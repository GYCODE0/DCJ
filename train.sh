

# train from MSMT17 to DukeMTMC-reID
python3 baseline_triplet+softmax_adam_adapt_cross-camera_duke.py
python3 selftraining_duke.py
python3 train_duke_softmax_lsr+triplet_adam_warmup.py

# train from MSMT17 to Market-1501
python3 baseline_triplet+softmax_adam_adapt_cross-camera_market.py
python3 selftraining_market.py
python3 train_market_softmax_lsr+triplet_adam_warmup.py


