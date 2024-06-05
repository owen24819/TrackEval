Cell-HOTA is an extention of HOTA. It is designed to handle cell divisions with all the benefits of HOTA. Althouhgh TrackEval has many metrics, the code was modified only to support the HOTA metric for cell tracking. All other metrics in TrackEval were not modified. This README.md was modified from the original TrackEval github.

HOTA was developed by Luiten et al. (*[HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking](https://link.springer.com/article/10.1007/s11263-020-01375-2). IJCV 2020. Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixe and Bastian Leibe.*)

Author of Cell-HOTA is Owen O'Connor (link to paper)

## Running the code

[here](scripts/run_cells_challenge.py) is the main script called run_cells_challenge.py. 

You need to set 4 variables to run it properly:

	dataset: str - the name of the dateset you are testing
 
  	model: str - the name of the model whose results you are testing
   
   	gt_path: Pathlib path - the path to the ground truth

	res_path: Pathlib path - the path to the model results

 There are 2 optional varialbes:

  	flex_div: bool (default: True) - determines whether Cell-HOTA allows flexible divisions
   
   	count_edges: bool (default: False) - determines whether Cell-HOTA uses cells touching the edge of the images towards the score
     
See each script for instructions and arguments.

## Properties of this codebase

The code is written 100% in python with only numpy and scipy as minimum requirements.

The code is designed to be easily understandable and easily extendable. 

The code is also extremely fast, running at more than 10x the speed of the both [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit), and [py-motmetrics](https://github.com/cheind/py-motmetrics) (see detailed speed comparison below).

The implementation of CLEARMOT and ID metrics aligns perfectly with the [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit).

By default the code prints results to the screen, saves results out as both a summary txt file and a detailed results csv file, and outputs plots of the results. All outputs are by default saved to the 'tracker' folder for each tracker.


## Quickly evaluate on supported benchmarks

To enable you to use TrackEval for evaluation as quickly and easily as possible, we provide ground-truth data, meta-data and example trackers for all currently supported benchmarks.
You can download this here: [data.zip](https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip) (~150mb).

The data for RobMOTS is separate and can be found here: [rob_mots_train_data.zip](https://omnomnom.vision.rwth-aachen.de/data/RobMOTS/train_data.zip) (~750mb).

The data for PersonPath22 is separate and can be found here: [person_path_22_data.zip](https://tracking-dataset-eccv-2022.s3.us-east-2.amazonaws.com/person_path_22_data.zip) (~3mb).

The easiest way to begin is to extract this zip into the repository root folder such that the file paths look like: TrackEval/data/gt/...

This then corresponds to the default paths in the code. You can now run each of the scripts [here](scripts/) without providing any arguments and they will by default evaluate all trackers present in the supplied file structure. To evaluate your own tracking results, simply copy your files as a new tracker folder into the file structure at the same level as the example trackers (MPNTrack, CIWT, track_rcnn, qdtrack, ags, Tracktor++, STEm_Seg), ensuring the same file structure for your trackers as in the example.

Of course, if your ground-truth and tracker files are located somewhere else you can simply use the script arguments to point the code toward your data.

To ensure your tracker outputs data in the correct format, check out our format guides for each of the supported benchmarks [here](docs), or check out the example trackers provided.

## Evaluate on your own custom benchmark

To evaluate on your own data, you have two options:
 - Write custom dataset code (more effort, rarely worth it).
 - Convert your current dataset and trackers to the same format of an already implemented benchmark.

To convert formats, check out the format specifications defined [here](docs).

By default, we would recommend the MOTChallenge format, although any implemented format should work. Note that for many cases you will want to use the argument ```--DO_PREPROC False``` unless you want to run preprocessing to remove distractor objects.

## Requirements
 Code tested on Python 3.7.
 
 - Minimum requirements: numpy, scipy
 - For plotting: matplotlib
 - For segmentation datasets (KITTI MOTS, MOTS-Challenge, DAVIS, YouTube-VIS): pycocotools
 - For DAVIS dataset: Pillow
 - For J & F metric: opencv_python, scikit_image
 - For simples test-cases for metrics: pytest

use ```pip3 -r install requirements.txt``` to install all possible requirements.

use ```pip3 -r install minimum_requirments.txt``` to only install the minimum if you don't need the extra functionality as listed above.

## License

TrackEval is released under the [MIT License](LICENSE).

## Citing TrackEval

If you use this code in your research, please use the following BibTeX entry:

```BibTeX
@misc{luiten2020trackeval,
  author =       {Jonathon Luiten, Arne Hoffhues},
  title =        {TrackEval},
  howpublished = {\url{https://github.com/JonathonLuiten/TrackEval}},
  year =         {2020}
}
```

Furthermore, if you use the HOTA metrics, please cite the following paper:

```
@article{luiten2020IJCV,
  title={HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking},
  author={Luiten, Jonathon and Osep, Aljosa and Dendorfer, Patrick and Torr, Philip and Geiger, Andreas and Leal-Taix{\'e}, Laura and Leibe, Bastian},
  journal={International Journal of Computer Vision},
  pages={1--31},
  year={2020},
  publisher={Springer}
}
```

If you use any other metrics please also cite the relevant papers, and don't forget to cite each of the benchmarks you evaluate on.
