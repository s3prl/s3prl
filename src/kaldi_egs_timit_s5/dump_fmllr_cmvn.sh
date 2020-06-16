for x in train dev test; do
  steps/compute_cmvn_stats.sh data-fmllr-tri3/$x exp/make_fmllr/$x data-fmllr-tri3/$x
done