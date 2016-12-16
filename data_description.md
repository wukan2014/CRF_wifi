## 数据结构
#####"config.mat":
###### Config:
- histo[voxel_num, stream_num* dim, states, n_bins ]: generated fingerprint of RSS data;
> voxel_num: number of grids;
> stream_num: number of wifi signal streams. In our experiments, we used 1 AP + 3 RX, thus stream_num = 3;
> dim: In CSI, subcarrier = 30, while for primary proof-of-concept, we compute the average RSS accross all subcarrier, in which case dim=1, we deduced CSI to RSS.
> states: ACE uses the cross-calibration way to model the CRF, RSS fingerprint is computed by P(s^t | alpha_x^t = 1) and P(s^t | alpha_i^t = 0, for i \neq x ). Thus states = 2 histograms are built.
> n_bins: singnal strength is constraint to [-30, 30] dBm, divided into n_bins = 120 bins
> the histograms was filtered by Gaussian filter.
- grid: In our experiments, the room is divided to 15*3 grids, while some grids are occupied by baffles, which targets could never be located at. Thus, these grids are not taken, denoted as "Invalid". '1' means valid grid, '-1' means baffle of cubicle(invalid).
```matlab
% grid = [-1, 1, -1, 1, -1, 1, -1; ...\
%         -1, 1, -1, 1, -1, 1, -1; ...\
%         -1, 1, -1, 1, -1, 1, -1; ...\
%         -1, 1, -1, 1, -1, 1, -1; ...\
%         -1, 1, -1, 1, -1, 1, -1; ...\
%         -1, 1, -1, 1, -1, 1, -1; ...\
%         -1, 1, -1, 1, -1, 1, -1; ...\
%         -1, 1, -1, 1, -1, 1, -1; ...\
%         -1, 1, -1, 1, -1, 1, -1; ...\
%         -1, 1, -1, 1, -1, 1, -1; ...\
%         -1, 1, -1, 1, -1, 1, -1; ...\
%         -1, 1, -1, 1, -1, 1, -1;...\
%         -1, 1, -1, 1, -1, 1, -1; ...\
%         1, 1, 1, 1, 1, 1, 1; ...\
%         1, 1, 1, 1, 1, 1, 1; ...\
%         1, 1, -1, -1, -1, 1, 1 ...\
% ];
```
- edges: BFS mapping grids to connected edges;
> "edges" is 2D list in dimension [num_edges, 2]. Each edge is denoted by [start_grid, end_grid]
    mapper: map voxel id back to grid index(row, col);

#### "samples.mat"
Train:
- seq[voxel_num*sample_len*2,stream_num]:
> concatenate RSS samples collected in different grids to a training sequence, '2' is for the static and dynamic target data collected in same voxel.
- label[voxel_num]: label of corresponding RSS sequence.
