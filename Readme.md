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

 There are 2 optional variables:

  	flex_div: bool (default: True) - determines whether Cell-HOTA allows flexible divisions
   
   	count_edges: bool (default: True) - determines whether Cell-HOTA uses cells touching the edge of the images towards the score
     
See each script for instructions and arguments.

## Evaluate on your own custom benchmark

To evaluate on your own data, convert your current dataset and trackers to the MOTS format.

To convert formats, check out the format specifications defined [here](docs).

## Requirements
 Code tested on Python 3.10.
 
 - Minimum requirements: numpy, scipy
 - For plotting: matplotlib
 - For segmentation datasets: pycocotools
 - For simples test-cases for metrics: pytest

use ```pip3 -r install requirements.txt``` to install all possible requirements.

## License

TrackEval is released under the [MIT License](LICENSE).

## Citing Cell-HOTA

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
