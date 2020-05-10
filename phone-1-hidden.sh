CUDA_VISIBLE_DEVICES=1 python runner_albert_downstream.py \
--train_phone \
--name "phone-epoch-40-log-1000-dev-10000-batch25" \
--epoch_train \
--run_mockingjay \
--only_query \
--config "config/mockingjay_libri_onlyQuery-1hidden-phone.yaml" \
--ckpdir "../newMockingjay/result_albert/albert-650000/ALBERT-6l-onlyQuery" \
--ckpt "mockingjay_libri_sd1337/mockingjayALBERT_variant-500000.ckpt" 

CUDA_VISIBLE_DEVICES=1 python runner_albert_downstream.py \
--test_phone \
--run_mockingjay \
--only_query \
--config "config/mockingjay_libri_onlyQuery-1hidden-phone.yaml" \
--ckpdir "../newMockingjay/result_albert/albert-650000/ALBERT-6l-onlyQuery" \
--ckpt "mockingjay_libri_sd1337/mockingjayALBERT_variant-500000.ckpt" \
--dckpt "phone-epoch-40-log-1000-dev-10000-batch25/best_val.ckpt"