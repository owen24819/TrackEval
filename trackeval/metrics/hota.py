
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mask_utils 
from ._base_metric import _BaseMetric
from .. import _timing


class HOTA(_BaseMetric):
    """Class which implements the HOTA metrics.
    See: https://link.springer.com/article/10.1007/s11263-020-01375-2
    """

    def __init__(self, flex_div=False, config=None):
        super().__init__()
        self.plottable = True
        self.array_labels = np.arange(0.05, 0.99, 0.05)
        self.integer_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP', 'Div_TP', 'Div_FP', 'Div_FN']
        self.float_array_fields = ['HOTA', 'DetA', 'AssA', 'DivA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'DivRe', 'DivPr', 'LocA', 'OWTA']
        self.float_fields = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
        self.error_fields = ['HOTA_FP_ID','HOTA_FN_ID','Div_FP_ID','Div_FN_ID','AssA_ID','Edges_GT_ID']
        # self.integer_fields = ['FP_pixel_counts','FN_pixel_counts']
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields
        self.summary_fields = self.float_array_fields + self.float_fields
        self.flex_div = flex_div

    @_timing.time
    def eval_sequence(self, data):
        """Calculates the HOTA metrics for one sequence"""

        # Initialise results
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=float)
        for field in self.float_fields:
            res[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['HOTA_FN'] = data['num_gt_dets'] * np.ones((len(self.array_labels)), dtype=float)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=float)
            res['LocA(0)'] = 1.0
            return res
        if data['num_gt_dets'] == 0:
            res['HOTA_FP'] = data['num_tracker_dets'] * np.ones((len(self.array_labels)), dtype=float)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=float)
            res['LocA(0)'] = 1.0
            return res

        # Variables counting global association
        potential_matches_count = np.zeros((data['num_gt_ids']+1, data['num_tracker_ids']+1)) # There is no cellID 0 so we add an extra empty spot
        gt_id_count = np.zeros((data['num_gt_ids']+1, 1))
        tracker_id_count = np.zeros((1, data['num_tracker_ids']+1))

        # First loop through each timestep and accumulate global track information.
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Count the potential matches between ids in each timestep
            # These are normalised, weighted by the match similarity.
            similarity = data['similarity_scores'][t]
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou

            # Calculate the total number of dets for each gt_id and tracker_id.
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        # Calculate overall jaccard alignment score (before unique matching) between IDs
        global_alignment_score = potential_matches_count / np.maximum(1,gt_id_count + tracker_id_count - potential_matches_count) # There is no cellID 0 so we ignore
        matches_counts = [np.zeros_like(potential_matches_count) for _ in self.array_labels]

        res['HOTA_FN_ID'] = [[[] for _ in range(len(self.array_labels))] for _ in range(len(data['gt_ids']))]
        res['HOTA_FP_ID'] = [[[] for _ in range(len(self.array_labels))] for _ in range(len(data['gt_ids']))]
        res['Div_FN_ID'] = [[[] for _ in range(len(self.array_labels))] for _ in range(len(data['gt_ids']))]
        res['Div_FP_ID'] = [[[] for _ in range(len(self.array_labels))] for _ in range(len(data['gt_ids']))]
        res['Edges_GT_ID'] = [[] for _ in range(len(data['gt_ids']))]

        res['AssA_ID'] = [[] for _ in range(len(self.array_labels))]
        data['FP_pixel_counts'] = 0
        data['FN_pixel_counts'] = 0

        check_for_flex_div_prev_frame = {'tracker':{},'gt':{}}

        # Calculate scores for each timestep
        for t, (gt_ids_t, tracker_ids_t, gt_parents_t, tracker_parents_t, edges_gt, edges_tracker) in enumerate(zip(data['gt_ids'], data['tracker_ids'], data['gt_parent'], data['tracker_parent'], data['edges_gt'], data['edges_tracker'])):

            # Here we keep track of late divisions to ensure the cell divides the next frame
            check_for_flex_div_fut_frame = {'tracker':{}, 'gt': {}}
            flex_matching = {'tracker':[], 'gt':[]}
            flex_tracker_ids = []
            flex_gt_ids = []

            # Deal with the case that there are no GT detections in a timestep.
            if len(gt_ids_t) == 0:
                tracker_ids_t = [tracker_id_t for tracker_id_t,edge in zip(tracker_ids_t,edges_tracker) if not edge]
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FP'][a] += len(tracker_ids_t)
                    res['HOTA_FP_ID'][t][a].extend(tracker_ids_t)# Keep track of cell IDs that are FP; This is just for display purposes
                continue

            # Deal with the case that there are no tracker detections in a timestep.
            if len(tracker_ids_t) == 0:
                gt_ids_t = [gt_id_t for gt_id_t,edge in zip(gt_ids_t,edges_gt) if not edge]
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FN'][a] += len(gt_ids_t)
                    res['HOTA_FN_ID'][t][a].extend(gt_ids_t)# Keep track of cell IDs taht are FN; This is just for display purposes
                continue

            # Get matching scores between pairs of dets for optimizing HOTA
            similarity = data['similarity_scores'][t]
            score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity

            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            match_rows_orig, match_cols_orig = match_rows, match_cols

            gt_ids_edges = gt_ids_t[edges_gt]
            gt_id_count[gt_ids_edges] -= 1
            num_edge_divs = 0

            # If a division happened where one cell is touching the edge, we ignore both
            gt_parent_edges = np.unique(gt_parents_t[edges_gt])
            gt_parent_edges = gt_parent_edges[gt_parent_edges!=-1]

            for gt_parent_edge in gt_parent_edges:
                gt_ids_edge = gt_ids_t[gt_parents_t == gt_parent_edge]
                
                for gt_id_edge in gt_ids_edge:
                    if gt_id_edge not in gt_ids_t[edges_gt]:
                        gt_id_count[gt_id_edge] -= 1
                        gt_ids_edges = np.concatenate((gt_ids_edges, np.array([gt_id_edge])))
                        edges_gt[gt_ids_t == gt_id_edge] = True
                        num_edge_divs += 1

            for gt_id in gt_ids_edges:
                gt_id_index = np.where(gt_ids_t == gt_id)[0]
                if gt_id_index in match_rows and similarity[gt_id_index,match_cols[match_rows == gt_id_index]] > 0:
                    tracker_id = tracker_ids_t[match_cols[match_rows == gt_id_index]][0]
                    tracker_id_count[0,tracker_id] -= 1
               
            # tracker_ids_edges = tracker_ids_t[edges_tracker]
            
            # for tracker_id in tracker_ids_edges:
            #     tracker_id_index = np.where(gt_ids_t == gt_id)[0]
            #     if tracker_id_index in match_cols and similarity[match_rows[match_cols == tracker_id_index],tracker_id_index] > 0:
            #         gt_id = gt_ids_t[match_rows[match_cols == gt_id_index]][0]
            #         gt_id_count[0,gt_id] -= 1                    

            # Remove cells that are touching the edge
            edges_match = edges_gt[match_rows] 
            match_cols = match_cols[~edges_match]
            match_rows = match_rows[~edges_match]

            # Get matched cells regardless of similarity score 
            gt_parents_match_t = gt_parents_t[match_rows]
            tracker_parents_match_t = tracker_parents_t[match_cols]

            # Count number of divisions for tracker and gt
            gt_divs = np.unique(gt_parents_t)
            gt_num_divs = len(gt_divs[gt_divs != -1])

            tracker_divs = np.unique(tracker_parents_t)
            tracker_num_divs = len(tracker_divs[tracker_divs != -1])

            # Get rid of divisions that occurred at the edge
            gt_div_edges = np.unique(gt_parents_t[edges_gt])
            gt_div_edges = gt_div_edges[gt_div_edges!=-1]

            for gt_div_edge in gt_div_edges:
                gt_div_edge_ind = np.where(gt_parents_t == gt_div_edge)[0]
                
                if gt_div_edge_ind[0] in match_rows_orig:
                    tracker_parent_edge = tracker_parents_t[match_cols_orig[match_rows_orig == gt_div_edge_ind[0]]]

                    if tracker_parent_edge != -1:
                         tracker_num_divs -= 1
                
                elif gt_div_edge_ind[1] in match_rows_orig: 
                    tracker_parent_edge = tracker_parents_t[match_cols_orig[match_rows_orig == gt_div_edge_ind[1]]]

                    if tracker_parent_edge != -1:
                         tracker_num_divs -= 1
                    
            gt_num_divs -= len(gt_div_edges)
            assert gt_num_divs >= 0 and tracker_num_divs >= 0 

            res['Edges_GT_ID'][t] = gt_ids_t[edges_gt]

            # Negative numbers below -10 indicate a flexible division; -11 is an arbitrary number below -1
            # When new_parent_id is used, a new one is generated by subtracting 1; so the next new_parent_id is -12
            # In case there are mulitple flexible divisions at one timepoint, the differnt parent_ids make it easy to match the cells
            new_parent_id = -11

            #Find which matching cells have a conflict where one cell just divided and the other tracked normally
            keep_mask_bool = np.ones(len(match_rows),dtype=bool)
            keep_track_bool = np.ones(len(match_rows),dtype=bool)
            flex_div_tracker_keep_remove_ind = np.zeros((0,2),dtype=int)
            flex_div_gt_keep_remove_ind = np.zeros((0,2),dtype=int)
            similarity_ind_dict = {}
            similarity_ind_dict['gt'] = {}
            similarity_ind_dict['tracker'] = {}

            if self.flex_div:

                # Get unique parent cell ids from each cell where division occurred for trackers and gts
                gt_unique_parents = np.unique(gt_parents_match_t)
                gt_unique_parents = gt_unique_parents[gt_unique_parents!=-1]
                tracker_unique_parents = np.unique(tracker_parents_match_t)
                tracker_unique_parents = tracker_unique_parents[tracker_unique_parents!=-1]

                # Iterate through parent cell ids of gts and check if a tracker cell is mismatched because it divdied late in the future frame at time t+1
                for gt_unique_parent in gt_unique_parents:

                    if gt_unique_parent in gt_parents_t[edges_gt]:
                        continue
                    
                    # Check if this cell qualified as a flexible division from the previous frame
                    if gt_unique_parent in check_for_flex_div_prev_frame['gt']:

                        tracker_parents_t[match_cols_orig[gt_parents_t[match_rows_orig] == gt_unique_parent]] = new_parent_id
                        gt_parents_t[match_rows_orig[gt_parents_t[match_rows_orig] == gt_unique_parent]] = new_parent_id

                        gt_parents_match_t = gt_parents_t[match_rows]
                        tracker_parents_match_t = tracker_parents_t[match_cols]

                        new_parent_id -= 1 
                        tracker_num_divs += 1


                    elif t+1 < len(data['gt_ids']): # flexible division can only occur if the next frame exists

                        # Get tracker cell id(s) that match to the parent cell id for gt
                        tracker_ids = tracker_ids_t[match_cols][gt_parents_match_t == gt_unique_parent]

                        # Iterate through tracker ids to see if a flexible division qualifies where the gt cell in frame t divdied but the tracker cell divdies in frame t+1
                        # Checking for flexible late divisions
                        for tracker_id in tracker_ids:

                            # If tracker and gt agree on cell division, there is no need for flexible division
                            # The GT parent cell divided at time t so we are checking if tracker cell had a late division at time t+1
                            #  Check that tracker did not just divide                # Check that cell divides in next frame          
                            if tracker_parents_t[tracker_ids_t == tracker_id] == -1 and tracker_id in data['tracker_parent'][t+1]: 
                                
                                # Get similarity scores from previous and future frame
                                similarity_prev = data['similarity_scores'][t-1]
                                similarity_fut = data['similarity_scores'][t+1]

                                # Get score_mat from previous and future frame
                                score_mat_prev = global_alignment_score[data['gt_ids'][t-1][:, np.newaxis], data['tracker_ids'][t-1][np.newaxis, :]] * similarity_prev
                                score_mat_fut = global_alignment_score[data['gt_ids'][t+1][:, np.newaxis], data['tracker_ids'][t+1][np.newaxis, :]] * similarity_fut

                                # Hungarian algorithm to find best matches for previous and future frame
                                match_rows_prev, match_cols_prev = linear_sum_assignment(-score_mat_prev)
                                match_rows_fut, match_cols_fut = linear_sum_assignment(-score_mat_fut)

                                # Check that tracker cell matches to gt cells in fut frame    # Check that tracker cell matched in prev frame
                                if tracker_id in data['tracker_parent'][t+1][match_cols_fut] and tracker_id in data['tracker_ids'][t-1][match_cols_prev]:

                                    # Get tracker cell id in the previous frame (t-1) that matched to the parent gt cell
                                    tracker_cell_id_prev = data['tracker_ids'][t-1][match_cols_prev][data['gt_ids'][t-1][match_rows_prev] == gt_unique_parent]
                                    # Get the gt cell ids in the future frame# Get the gt cell ids in the future frame (t+1) that matched to the tracker cell post division (t+1) that matched to the tracker cell post division
                                    gt_cell_ids_fut = data['gt_ids'][t+1][match_rows_fut][data['tracker_parent'][t+1][match_cols_fut] == tracker_id]

                                    # Check if gt cell divides again in future frame; if cell divides two frames in a row, we disqualify it for flexible division
                                    gt_divided_in_fut_frame = data['gt_parent'][t+1][match_rows_fut][data['tracker_parent'][t+1][match_cols_fut] == tracker_id].sum() != -2

                                    # Get indexes of gt cells that divided for gt parent id of interest 
                                    div_cells_ind = np.where(gt_parents_t == gt_unique_parent)[0]
                                    # Get gt cell ids that divided for gt parent id of interest 
                                    gt_div_cells_ids = gt_ids_t[div_cells_ind] # Two cells ids 

                                    fut_edges = data['edges_gt'][t+1]
                                    fut_gt_ids = data['gt_ids'][t+1]
                                    fut_gt_ids_edges = fut_gt_ids[fut_edges]
                                    fut_gt_ids_flex_edge = gt_cell_ids_fut[0] in fut_gt_ids_edges or (len(gt_cell_ids_fut) > 1 and gt_cell_ids_fut[1] in fut_gt_ids_edges)
                                    
                                    # Check that the tracker cell matches wtih the parent gt cell in previous frame (t-1)
                                    # Check that the divided tracker cells match with the divided gt cells in frame (t+1)
                                    if np.array_equal(gt_div_cells_ids,np.sort(gt_cell_ids_fut)) and tracker_cell_id_prev == tracker_id and not gt_divided_in_fut_frame and not fut_gt_ids_flex_edge:
                                        
                                        # Combine rles of gt cells that divided at time t (that correspond to the parent gt cell id of interest)
                                        gt_1_rle = data['gt_dets'][t][div_cells_ind[0]]
                                        gt_1_mask = mask_utils.decode(gt_1_rle) # convert rle to np array
                                        gt_2_rle = data['gt_dets'][t][div_cells_ind[1]]
                                        gt_2_mask = mask_utils.decode(gt_2_rle) # convert rle to np array
                                        gt_comb_mask = gt_1_mask + gt_2_mask # combine divdided cells into one np array
                                        assert gt_comb_mask.max() == 1
                                        gt_comb_rle = mask_utils.encode(gt_comb_mask) # convert np array of divided cells into rle
                                        assert div_cells_ind.__len__() == 2

                                        # Get rle for tracker cell that matches to one of the gt cells and will divide at time t+1
                                        tracker_ind = np.where((gt_parents_match_t == gt_unique_parent) * (tracker_parents_match_t == -1))[0][0]
                                        tracker_ind = np.where(tracker_ids_t == tracker_id)[0][0]
                                        tracker_rle = data['tracker_dets'][t][tracker_ind]

                                        # Get IOU between divided gt cells and tracker cell at time t
                                        iou = mask_utils.iou([gt_comb_rle],[tracker_rle],[False])

                                        # Save in dictionary which tracker cell and gt cells are saved for flexible division
                                        check_for_flex_div_fut_frame['tracker'][tracker_ids_t[tracker_ind]] = gt_ids_t[div_cells_ind]

                                        # Currently, only 1 of the gt cells are matched to the correct tracker cell when both gt cells should be matched to the tracker cell
                                        # We need to create one matching between the two gt cells and one tracker cell
                                        # In order to do this, we keep one matching and remove the other if it exists
                                        
                                        keep_gt_ind = match_rows[match_cols == tracker_ind][0]
                                        remove_gt_ind = div_cells_ind[0] if div_cells_ind[1] == keep_gt_ind else div_cells_ind[1]

                                        # Replace iou between gt cell and tracker cell with combined iou between gt cells and tracker cell
                                        similarity[keep_gt_ind,tracker_ind] = iou
                                        similarity_ind_dict['tracker'][gt_div_cells_ids[0]] = [keep_gt_ind,tracker_ind]
                                        similarity_ind_dict['tracker'][gt_div_cells_ids[1]] = [keep_gt_ind,tracker_ind]

                                        # Remove unused indice if it was used in matching and make sure used indice is being used
                                        keep_mask_bool *= match_rows != remove_gt_ind # make match null if remove ind matched to different tracker cell

                                        # For tracking, we ignore flexible divisions all together since they will be assessed separately
                                        keep_track_bool *= (match_rows != keep_gt_ind) * (match_rows != remove_gt_ind)

                                        # We remove cell division for the gt cell for this frame and will a cell division for the gt cell in the next frame
                                        gt_num_divs -= 1

                                        # keep track of indices of gt cells that were kept / removed - not used in HOTA calculation - just used to display FN / FPs
                                        flex_div_gt_keep_remove_ind = np.concatenate((flex_div_gt_keep_remove_ind,np.array([[keep_gt_ind, remove_gt_ind]])))

                                        # Keep track of tracker ids that were used for flexible division
                                        flex_tracker_ids.append(tracker_id)

                                        # gt_id_count and tracker_id_count is used to calculate AssA, AssRe & AssPr 
                                        # During a flexible late division, simulate two divided tracker cells existing at time t instead of time t+1 instead of the original mother tracker cell exisiting at time t
                                        tracker_id_count[0,tracker_id] -= 1
                                        
                                        # Here we add to tracker_id_count to match the gt_id_count
                                        tracker_cell_ids_fut = data['tracker_ids'][t+1][match_cols_fut][data['tracker_parent'][t+1][match_cols_fut] == tracker_id]
                                        assert len(tracker_cell_ids_fut) == 2
                                        tracker_id_count[0,tracker_cell_ids_fut] += 1

                                        gt_id_0 = data['gt_ids'][t+1][match_rows_fut][data['tracker_ids'][t+1][match_cols_fut] == tracker_cell_ids_fut[0]][0]
                                        gt_id_1 = data['gt_ids'][t+1][match_rows_fut][data['tracker_ids'][t+1][match_cols_fut] == tracker_cell_ids_fut[1]][0]
                                        flex_matching['tracker'].append([gt_id_0,tracker_cell_ids_fut[0],])
                                        flex_matching['tracker'].append([gt_id_1,tracker_cell_ids_fut[1]])

                # Iterate through parent cell ids of trackers
                # Checking for flexible early divisions
                for tracker_unique_parent in tracker_unique_parents:                  

                    if tracker_unique_parent in check_for_flex_div_prev_frame['tracker']:

                        # gt_parents_t[match_rows[tracker_parents_match_t == tracker_unique_parent]] = new_parent_id
                        # tracker_parents_t[match_cols[tracker_parents_match_t == tracker_unique_parent]] = new_parent_id

                        gt_parents_prev = data['gt_parent'][t-1]
                        gt_ids_prev = data['gt_ids'][t-1]

                        gt_id = gt_ids_t[match_rows[tracker_parents_match_t == tracker_unique_parent]][0]

                        gt_parent = gt_parents_prev[gt_ids_prev == gt_id][0]
                        gt_ids = gt_ids_prev[gt_parents_prev == gt_parent]

                        for gt_id in gt_ids:
                            gt_parents_t[gt_ids_t == gt_id] = new_parent_id

                        tracker_parents_t[tracker_parents_t == tracker_unique_parent] = new_parent_id

                        gt_parents_match_t = gt_parents_t[match_rows]
                        tracker_parents_match_t = tracker_parents_t[match_cols]

                        new_parent_id -= 1
                        gt_num_divs += 1

                    elif t+1 < len(data['gt_ids']):
                        gt_ids = gt_ids_t[match_rows][tracker_parents_match_t == tracker_unique_parent]

                        for gt_id in gt_ids:

                            if gt_id in gt_ids_t[edges_gt]:
                                continue

                            # If tracker and gt agree on cell division, there is no need for flexible division
                            # Check that tracker did not just divided   # Check that cell divides in next frame
                            if gt_parents_t[gt_ids_t == gt_id] == -1 and gt_id in data['gt_parent'][t+1]:

                                # Get similarity scores from previous and future frame
                                similarity_prev  = data['similarity_scores'][t-1]
                                similarity_fut  = data['similarity_scores'][t+1]

                                # Get score_mat from previous and future frame
                                score_mat_prev = global_alignment_score[data['gt_ids'][t-1][:, np.newaxis], data['tracker_ids'][t-1][np.newaxis, :]] * similarity_prev
                                score_mat_fut = global_alignment_score[data['gt_ids'][t+1][:, np.newaxis], data['tracker_ids'][t+1][np.newaxis, :]] * similarity_fut

                                # Hungarian algorithm to find best matches
                                match_rows_prev, match_cols_prev = linear_sum_assignment(-score_mat_prev)
                                match_rows_fut, match_cols_fut = linear_sum_assignment(-score_mat_fut)

                                # Get gt cell id in the previous frame (t-1) that matched to the parent tracker cell
                                match_div_cells_prev = data['gt_ids'][t-1][match_rows_prev][data['tracker_ids'][t-1][match_cols_prev] == tracker_unique_parent]
                                # Get the tracker cell ids in the future frame (t+1) that matched to the gt cell post division
                                match_div_cells_fut = data['tracker_ids'][t+1][match_cols_fut][data['gt_parent'][t+1][match_rows_fut] == gt_id]

                                # Check if tracker cell divides again in future frame; if cell divides two frames in a row, we disqualify it for flexible division
                                tracker_divided_in_fut_frame = data['tracker_parent'][t+1][match_cols_fut][data['gt_parent'][t+1][match_rows_fut] == gt_id].sum() != -2
                                
                                # Get indexes of tracker cells that divided for tracker parent id of interest 
                                div_cells_ind = np.where(tracker_parents_t == tracker_unique_parent)[0]
                                # Get tracker cell ids that divided for tracker parent id of interest 
                                tracker_div_cells_ids = tracker_ids_t[div_cells_ind]

                                # Check that when tracker cell divided in frame t+1, they will match with the gt cells
                                if np.array_equal(tracker_div_cells_ids,np.sort(match_div_cells_fut)) and match_div_cells_prev == gt_id and not tracker_divided_in_fut_frame:

                                    # Combine rles of tracker cells that divided at time t (that correspond to the parent tracker cell id of interest)
                                    tracker_1_rle = data['tracker_dets'][t][div_cells_ind[0]]
                                    tracker_1_mask = mask_utils.decode(tracker_1_rle) # convert rle to np array
                                    tracker_2_rle = data['tracker_dets'][t][div_cells_ind[1]]
                                    tracker_2_mask = mask_utils.decode(tracker_2_rle) # convert rle to np array
                                    tracker_comb_mask = tracker_1_mask + tracker_2_mask # combine divdided cells into one np array
                                    assert tracker_comb_mask.max() == 1
                                    tracker_comb_rle = mask_utils.encode(tracker_comb_mask) # convert np array of divided cells into rle
                                    assert div_cells_ind.__len__() == 2

                                    # Get rle for gt cell that matches to one of the tracker cells and will divide at time t+1
                                    gt_ind = np.where(gt_ids_t == gt_id)[0][0]
                                    gt_rle = data['gt_dets'][t][gt_ind]

                                    # Get IOU between divided gt cells and tracker cell at time t
                                    iou = mask_utils.iou([tracker_comb_rle],[gt_rle],[False])

                                    # Save in dictionary which tracker cell and gt cells are saved for flexible division
                                    check_for_flex_div_fut_frame['gt'][gt_id] = tracker_ids_t[div_cells_ind]
                                    assert gt_id == gt_ids_t[gt_ind]

                                    keep_tracker_ind = match_cols[match_rows == gt_ind][0]
                                    remove_tracker_ind = div_cells_ind[0] if div_cells_ind[1] == keep_tracker_ind else div_cells_ind[1]

                                    # Replace iou between gt cell and tracker cell with combined iou between gt cells and tracker cell
                                    similarity[gt_ind,keep_tracker_ind] = iou

                                    similarity_ind_dict['gt'][gt_id] = [gt_ind,keep_tracker_ind]

                                    # We have one value in similarity that repersents the IOU overlap between the two tracker cells and
                                    keep_mask_bool *= match_cols != remove_tracker_ind # make match null if remove ind matched to different tracker cell

                                    # For tracking, we ignore flexible divisions
                                    keep_track_bool *= (match_cols != keep_tracker_ind) * (match_cols != remove_tracker_ind)

                                    # We remove cell division for the tracker cell in this frame and add it to the next frame at time t+1
                                    tracker_num_divs -= 1

                                    # keep track of which indices were kept and removed - used to display FN / FPs
                                    flex_div_tracker_keep_remove_ind  = np.concatenate((flex_div_tracker_keep_remove_ind, np.array([[keep_tracker_ind, remove_tracker_ind]])))

                                    # Keep track of tracker ids that were used for flexible division
                                    flex_gt_ids.append(gt_id)

                                    # gt_id_count and tracker_id_count is used to calculate AssA, AssRe & AssPr 
                                    # During a flexible division, we match the flexible division to the gt
                                    tracker_id_count[0,tracker_div_cells_ids] -= 1
                                    parent_tracker_id = tracker_parents_t[tracker_ids_t == tracker_div_cells_ids[0]][0]
                                    tracker_id_count[0,parent_tracker_id] += 1

                                    flex_matching['gt'].append([gt_id,parent_tracker_id])

            check_for_flex_div_prev_frame = check_for_flex_div_fut_frame.copy()

            # We ignore flexible divisions for tracking
            match_rows_track = match_rows[keep_track_bool]
            match_cols_track = match_cols[keep_track_bool]

            # We keep flexible divisions here but remove matches where secondary matches were made due to flexible divisions
            match_rows_mask = match_rows[keep_mask_bool]
            match_cols_mask = match_cols[keep_mask_bool]

            gt_dets_t_masks = np.array([mask_utils.decode(mask) for mask in data['gt_dets'][t]])[match_rows_mask]
            tracker_dets_t_masks = np.array([mask_utils.decode(mask) for mask in data['tracker_dets'][t]])[match_cols_mask]

            # Keep track of FP / FN at a high levels; helps determine if algorithm is under or over segmenting; not used to calculate HOTA
            FP = ((1-gt_dets_t_masks) * tracker_dets_t_masks).sum()
            FN = (gt_dets_t_masks * (1-tracker_dets_t_masks)).sum()
            data['FP_pixel_counts'] += FP
            data['FN_pixel_counts'] += FN

            # Figure out which dividing cells match
            # We need to make sure that a set of two tracker and two gt cells just divided at time and have parents that match as well
            # Then the combined IOU between the tracker and gt cells will be used to determine if it is cell division (depending on alpha)

            # Get matches where both matching cells divded at time t
            div_matches = (gt_parents_t[match_rows_track] != -1) * (tracker_parents_t[match_cols_track] != -1)

            # If there is at least one set of matching cells that divide at time t
            if div_matches.sum() > 0:

                # First we ensure that each matching set has two cells; if a cell that divided matches to a cell that doesn't divide, we discard it
                div_match_gt_parents = gt_parents_t[match_rows_track][div_matches]

                # If a gt cell divided at time t and only one daughter gt cell matched to a tracker cell then we can discard this match 
                for div_match_gt_parent in np.unique(div_match_gt_parents):
                    if (div_match_gt_parents == div_match_gt_parent).sum() != 2:
                        div_matches[gt_parents_t[match_rows_track] == div_match_gt_parent] = False

                div_match_tracker_parents = tracker_parents_t[match_cols_track][div_matches]

                # If a tracker cell divided at time t and only one daughter tracker cell matched to a gt cell then we can discard this match 
                for div_match_tracker_parent in np.unique(div_match_tracker_parents):
                    if (div_match_tracker_parents == div_match_tracker_parent).sum() != 2:
                        div_matches[tracker_parents_t[match_cols_track] == div_match_tracker_parent] = False

                # At this point, all sets of matching cells contain two cells both divided; Next need to check if they have matching parents
                if div_matches.sum() > 0:
                    # Get all parent ids of matching cells; We use match_rows_track to remove cells that were used for flexible division for time t
                    # Cell involved in a flexilbe division from time t-1 will be included in match_rows_track and the parent cell id will be a number below as -10 as an easy way to identify
                    gt_parents_match_t_div = gt_parents_t[match_rows_track][div_matches]

                    # Get unique parent cell ids for gt and tracker cells that divided
                    gt_parents_match_t_div_unique = np.unique(gt_parents_match_t_div)

                    # Get matching gt and tracker ids from the previous frame (time t-1)
                    gt_prev_ids_match = data['gt_ids'][t-1][prev_matches[0]]                        
                    tracker_prev_ids_match = data['tracker_ids'][t-1][prev_matches[1]]

                    # Iterate through gt parent 
                    for gt_parent_match_t_div in gt_parents_match_t_div_unique:

                        if gt_parent_match_t_div > -10: # This signifies a flexible division from previous frame (time t-1) which we can ignore because this is already confirmed as a correct division

                            # Get tracker parent cell id that is matched with gt parent cell id
                            tracker_parent_match_t_div = tracker_parents_match_t[gt_parents_match_t == gt_parent_match_t_div][0]

                            # Check if parent gt and tracker cells match
                            parents_of_div_cells_match = gt_prev_ids_match[tracker_prev_ids_match == tracker_parent_match_t_div] == gt_parent_match_t_div
                            
                            if not parents_of_div_cells_match:
                                div_matches[gt_parents_t[match_rows_track] == gt_parent_match_t_div] = False

                    # If matches are still left, calculate IOU overlap
                    if div_matches.sum() > 0:

                        # Get IOU of cell divisions and reshape the array into Num_of_divs x 2
                        similarity_div = similarity[match_rows_track,match_cols_track][div_matches]
                        assert similarity_div.shape[0] % 2 == 0
                        similarity_div = np.array(np.split(similarity_div,similarity_div.shape[0] // 2))
                            
                        # Get the geometric mean of the iou between the two divided cells 
                        iou_div = np.sqrt(np.prod(similarity_div,-1)) 

            num_trackers_match_to_gt_edges = (similarity[match_rows_orig[edges_match],match_cols_orig[edges_match]] > 0).sum()

            # Calculate and accumulate basic statistics
            for a, alpha in enumerate(self.array_labels):
                num_late_divs = 0
                num_early_divs = 0

                actually_matched_mask = similarity[match_rows_mask, match_cols_mask] >= alpha - np.finfo('float').eps
                alpha_match_rows = match_rows_mask[actually_matched_mask]
                alpha_match_cols = match_cols_mask[actually_matched_mask]
                num_matches = len(alpha_match_rows)

                # If a late division happens and the IOU (1 tracker and 2 gt) is above the threshold alpha, we mark it as a sucessful late division
                # We also add an extra match since there are 2 gt cells
                if len(flex_tracker_ids) > 0:
                    for flex_tracker_id in flex_tracker_ids:
                        if flex_tracker_id in tracker_ids_t[alpha_match_cols]:
                            num_late_divs += 1
                            num_matches += 1

                # If a early division happens and the IOU (2 tracker and 1 gt) is above the threshold alpha, we mark it as a sucessful early division
                # We don't add an extra match since there is 1 gt cell
                if len(flex_gt_ids) > 0:
                    for flex_gt_id in flex_gt_ids:
                        if flex_gt_id in gt_ids_t[alpha_match_rows]:
                            num_early_divs += 1

                # Early and late flexible divisions do not change the HOTA score vs getting the ground truth correct without a flexible division
                res['HOTA_TP'][a] += num_matches
                num_gt_ids = len(gt_ids_t) - edges_gt.sum()
                FNs_t = num_gt_ids - num_matches
                res['HOTA_FN'][a] += FNs_t
                num_tracker_ids = len(tracker_ids_t) + num_late_divs - num_early_divs - num_trackers_match_to_gt_edges
                FPs_t = num_tracker_ids - num_matches
                res['HOTA_FP'][a] += FPs_t

                assert FNs_t >= 0 and FPs_t >= 0

                # Save IDs of cells identified as FNs; This is just for display purposes
                if FNs_t > 0:

                    # Get gt ids that were False for alpha_match_rows
                    FN_index = np.array([i for i in range(len(gt_ids_t)) if i not in alpha_match_rows and not edges_gt[i]])

                    # If flexible division was above threshold alpha, we have to remove the flex match that was discarded
                    for gt_keep_ind in flex_div_gt_keep_remove_ind[:,0]:
                        if gt_keep_ind not in FN_index:
                            gt_remove_ind = flex_div_gt_keep_remove_ind[flex_div_gt_keep_remove_ind[:,0] == gt_keep_ind,1][0]
                            FN_index = FN_index[FN_index != gt_remove_ind]

                    res['HOTA_FN_ID'][t][a] = np.atleast_1d(gt_ids_t[FN_index])

                assert len(res['HOTA_FN_ID'][t][a]) == FNs_t

                # Save IDs of cells identified as FPs; This is just for display purposes
                if FPs_t > 0:

                    # Get gt ids that were False for alpha_match_rows
                    FP_index = np.array([i for i in range(len(tracker_ids_t)) if i not in alpha_match_cols and (i not in match_cols_orig or (not edges_match[match_cols_orig == i] or (edges_match[match_cols_orig == i] and similarity[match_rows_orig[match_cols_orig==i],i] == 0)))])

                    # If flexible division was above threshold alpha, we have to remove the flex match that was discarded
                    for tracker_keep_ind in flex_div_tracker_keep_remove_ind[:,0]:
                        if tracker_keep_ind not in FP_index:
                            tracker_remove_ind = flex_div_tracker_keep_remove_ind[flex_div_tracker_keep_remove_ind[:,0] == tracker_keep_ind,1][0]
                            FP_index = FP_index[FP_index != tracker_remove_ind]

                    res['HOTA_FP_ID'][t][a] = np.atleast_1d(tracker_ids_t[FP_index])

                assert len(res['HOTA_FP_ID'][t][a]) == FPs_t

                if num_matches > 0:
                    res['LocA'][a] += sum(similarity[alpha_match_rows, alpha_match_cols])

                    # If there was a flexible division, we will remove it from consideration for tracking / division but still count towards object detection
                    actually_matched_track = similarity[match_rows_track, match_cols_track] >= alpha - np.finfo('float').eps
                    alpha_match_rows = match_rows_track[actually_matched_track]
                    alpha_match_cols = match_cols_track[actually_matched_track]

                    matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1

                    # For flexible late division, we simulate tracker cell dividing at time t; we add two connections between the tracker cells in time t+1 with the gt cells in time t
                    if len(flex_matching['tracker']) > 0:
                        for (gt_id,tracker_id) in flex_matching['tracker']:
                            row, col = similarity_ind_dict['tracker'][gt_id]
                            if similarity[row,col] > alpha:
                                matches_counts[a][gt_id, tracker_id] += 1

                    # For early division, we match the gt for time t; we add one connection between the tracker cell in time t-1 with the gt cell in time t
                    if len(flex_matching['gt']) > 0:
                        for (gt_id,tracker_id) in flex_matching['gt']:
                            row, col = similarity_ind_dict['gt'][gt_id]
                            if similarity[row,col] > alpha:
                                matches_counts[a][gt_id, tracker_id] += 1

                num_div_matches = 0

                if gt_num_divs == 0 and tracker_num_divs > 0:
                    res['Div_FP'][a] += tracker_num_divs
                elif tracker_num_divs == 0 and gt_num_divs > 0:
                    res['Div_FN'][a] += gt_num_divs

                elif gt_num_divs > 0 and tracker_num_divs > 0:
                    
                    if div_matches.sum() > 0:
                        # Use parent_div_matches to remove cells that don't have matching parent cells at time t-1 and use alpha as threshold to determine if the cell division is a match
                        div_matches_alpha = iou_div > alpha

                        # Add up total number of matches where a cell division occured
                        num_div_matches = (div_matches_alpha).sum()

                    res['Div_TP'][a] += num_div_matches
                    res['Div_FN'][a] += gt_num_divs - num_div_matches
                    res['Div_FP'][a] += tracker_num_divs - num_div_matches

                    assert gt_num_divs - num_div_matches >= 0 and tracker_num_divs - num_div_matches >= 0 and num_div_matches >= 0

                gt_parents_t_edges = np.unique(gt_parents_t[edges_gt])
                gt_parents_t_edges = gt_parents_t_edges[gt_parents_t_edges != -1]

                # Save cell ID where a FN cell division occurred; This is just for display purposes
                if gt_num_divs - num_div_matches > 0:

                    # Get unique gt parent ids
                    gt_parents_t_unique = np.unique(gt_parents_t)
                    gt_parents_t_unique = gt_parents_t_unique[gt_parents_t_unique != -1]

                    for gt_parents_t_edge in gt_parents_t_edges:
                        gt_parents_t_unique = gt_parents_t_unique[gt_parents_t_unique != gt_parents_t_edge]

                    gt_parents_t_match_unique = gt_parents_t[match_rows_track][div_matches]
                    gt_parents_t_match_unique = np.unique(gt_parents_t_match_unique)

                    # Record the cell ids that are FNs for cell division
                    for gt_parent_t_unique in gt_parents_t_unique:
                        # If cell is involved in a flexible division for time t, we ignore it and assess it in the next frame at time t+1
                        if np.where(gt_parents_t == gt_parent_t_unique)[0][0] in flex_div_gt_keep_remove_ind:
                            continue
                        # we check that gt_parent_t_unique not a successful match
                        if num_div_matches == 0 or gt_parent_t_unique not in gt_parents_t_match_unique or not div_matches_alpha[np.where(gt_parents_t_match_unique == gt_parent_t_unique)[0][0]]:
                            res['Div_FN_ID'][t][a].extend(gt_ids_t[gt_parents_t == gt_parent_t_unique]) 

                assert gt_num_divs - num_div_matches == len(res['Div_FN_ID'][t][a]) // 2

                # Save cell ID where a FP cell division occurred; This is just for display purposes
                if tracker_num_divs - num_div_matches > 0:
                    
                    # Get unique tracker parent ids
                    tracker_parents_t_unique = np.unique(tracker_parents_t)
                    tracker_parents_t_unique = tracker_parents_t_unique[tracker_parents_t_unique != -1]

                    for gt_parents_t_edge in gt_parents_t_edges:

                        tracker_parent_edges = []

                        gt_div_edge_ind = np.where(gt_parents_t == gt_parents_t_edge)[0]
                        
                        if gt_div_edge_ind[0] in match_rows_orig:
                            tracker_parent_edge = tracker_parents_t[match_cols_orig[match_rows_orig == gt_div_edge_ind[0]]]
                            if tracker_parent_edge != -1:
                                tracker_parent_edges.append(tracker_parent_edge)

                        elif gt_div_edge_ind[1] in match_rows_orig: 
                            tracker_parent_edge = tracker_parents_t[match_cols_orig[match_rows_orig == gt_div_edge_ind[1]]]
                            if tracker_parent_edge != -1:
                                tracker_parent_edges.append(tracker_parent_edge)

                        for tracker_parent_edge in tracker_parent_edges:
                            tracker_parents_t_unique = tracker_parents_t_unique[tracker_parents_t_unique != tracker_parent_edge]

                    tracker_parents_t_match_unique = tracker_parents_t[match_cols_track][div_matches]
                    tracker_parents_t_match_unique = np.unique(tracker_parents_t_match_unique)

                    # Record the cell ids that are FNs for cell division
                    for tracker_parent_t_unique in tracker_parents_t_unique:
                        # If cell is involved in a flexible division for time t, we ignore it and assess it in the next frame at time t+1
                        if np.where(tracker_parents_t == tracker_parent_t_unique)[0][0] in flex_div_tracker_keep_remove_ind:
                            continue
                        # we check that tracker_parent_t_unique not a successful match
                        if num_div_matches == 0 or tracker_parent_t_unique not in tracker_parents_t_match_unique or not div_matches_alpha[np.where(tracker_parents_t_match_unique == tracker_parent_t_unique)[0][0]]:
                            res['Div_FP_ID'][t][a].extend(tracker_ids_t[tracker_parents_t == tracker_parent_t_unique]) 

                assert tracker_num_divs - num_div_matches == len(res['Div_FP_ID'][t][a]) // 2

            prev_matches = [match_rows, match_cols]

        # Calculate association scores (AssA, AssRe, AssPr) for the alpha value.
        # First calculate scores per gt_id/tracker_id combo and then average over the number of detections.
        for a, alpha in enumerate(self.array_labels):
            matches_count = matches_counts[a]
            assert (matches_count > gt_id_count).sum() + (matches_count > tracker_id_count).sum() == 0
            ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
            res['AssA'][a] = np.sum(matches_count * ass_a) / np.maximum(1, res['HOTA_TP'][a])
            ass_re = matches_count / np.maximum(1, gt_id_count)
            res['AssRe'][a] = np.sum(matches_count * ass_re) / np.maximum(1, res['HOTA_TP'][a])
            ass_pr = matches_count / np.maximum(1, tracker_id_count)
            res['AssPr'][a] = np.sum(matches_count * ass_pr) / np.maximum(1, res['HOTA_TP'][a])

            error_ind = np.where((ass_a > 0)*(ass_a < 1))

            res['AssA_ID'][a] = [error_ind,ass_a[error_ind]]

        # Calculate final scores
        res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')
        # for field in self.integer_fields:
        #     res[field] = self._combine_sum(all_res, field)
        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        for field in self.integer_array_fields:
            if ignore_empty_classes:
                res[field] = self._combine_sum(
                    {k: v for k, v in all_res.items()
                     if (v['HOTA_TP'] + v['HOTA_FN'] + v['HOTA_FP'] > 0 + np.finfo('float').eps).any()}, field)
            else:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()}, field)

        for field in self.float_fields + self.float_array_fields:
            if ignore_empty_classes:
                res[field] = np.mean([v[field] for v in all_res.values() if
                                      (v['HOTA_TP'] + v['HOTA_FN'] + v['HOTA_FP'] > 0 + np.finfo('float').eps).any()],
                                     axis=0)
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')
        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])

        res['DivRe'] = res['Div_TP'] / np.maximum(1, res['Div_TP'] + res['Div_FN'])
        res['DivPr'] = res['Div_TP'] / np.maximum(1, res['Div_TP'] + res['Div_FP'])
        res['DivA'] = res['Div_TP'] / np.maximum(1, res['Div_TP'] + res['Div_FN'] + res['Div_FP'])

        res['HOTA'] = np.sqrt(res['DetA'] * np.sqrt(res['AssA'] * res['DivA']))
        res['OWTA'] = np.sqrt(res['DetRe'] * np.sqrt(res['AssA'] * res['DivA']))

        res['HOTA(0)'] = res['HOTA'][0]
        res['LocA(0)'] = res['LocA'][0]
        res['HOTALocA(0)'] = res['HOTA(0)']*res['LocA(0)']
        return res

    def plot_single_tracker_results(self, table_res, tracker, cls, output_folder):
        """Create plot of results"""

        # Only loaded when run to reduce minimum requirements
        from matplotlib import pyplot as plt

        res = table_res['COMBINED_SEQ']
        styles_to_plot = ['r', 'b', 'g', 'm', 'c', 'orange', 'lime', 'silver', 'k']
        colostyles_to_plotrs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#17becf']

        fields_plot_names = [
                        [['HOTA','DetA','AssA','DivA','LocA'],'Overall'],
                        [['DetA','DetPr','DetRe','AssA','AssPr','AssRe','DivA','DivPr','DivRe'],'Precision_Recall'],
                        [['DetA','DetPr','DetRe'],'Det'],
                        [['AssA','AssPr','AssRe'],'Ass'],
                        [['DivA','DivPr','DivRe'],'Div'],

        ]
        for fields, plot_name in fields_plot_names:
            for name, style in zip(fields, styles_to_plot):
                plt.plot(self.array_labels, res[name], style)
            plt.xlabel('alpha')
            plt.ylabel('score')
            plt.title(tracker + ' - ' + cls)
            plt.axis([0, 1, 0, 1])
            legend = []
            for name in fields:
                legend += [name + ' (' + str(np.round(np.mean(res[name]), 2)) + ')']
            plt.legend(legend, loc='lower left')
            flex_div = '_flex_div' if self.flex_div else ''
            out_file = os.path.join(output_folder, cls + '_' + plot_name + flex_div + '_plot.pdf')
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            plt.savefig(out_file)
            plt.savefig(out_file.replace('.pdf', '.png'))
            plt.clf()