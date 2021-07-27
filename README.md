# DCJ
code for A Three-stage Learning Approach to Cross-Domain Person Re-identification

environment:
python = 3.6
torch = 1.0

download and unzip Market-1501-v15.09.15, DukeMTMC-reID, MSMT17 to /data

run train.sh

# train from MSMT17 to DukeMTMC-reID
python3 baseline_triplet+softmax_adam_adapt_cross-camera_duke.py # domain and camera adaptation
python3 selftraining_duke.py # self-supervised clustering re-training
python3 train_duke_softmax_lsr+triplet_adam_warmup.py #  joint loss learning

# train from MSMT17 to Market-1501
python3 baseline_triplet+softmax_adam_adapt_cross-camera_market.py #  domain and camera adaptation
python3 selftraining_market.py # self-supervised clustering re-training
python3 train_market_softmax_lsr+triplet_adam_warmup.py #  joint loss learning
