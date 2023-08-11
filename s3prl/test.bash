echo "Concat_dimension\n" > result.txt
python3 run_downstream_concat_dim.py -m train -n wavlm_and_hubert_concat_dim_1h -u hubert -d asr -o "config.downstream_expert.datarc.train=['1h']"
echo "\ndev:\n" > result.txt
python3 run_downstream_concat_dim.py -m evaluate -e result/downstream/wavlm_and_hubert_concat_dim_1h/dev-clean-best.ckpt -t 'dev-clean' | tail -n2 >> result.txt
echo "\n#################################\n" > result.txt
echo "\ntest:\n" > result.txt
python3 run_downstream_concat_dim.py -m evaluate -e result/downstream/wavlm_and_hubert_concat_dim_1h/dev-clean-best.ckpt -t 'test-clean' | tail -n2 >> result.txt
echo "\nAttention\n" > result.txt
python3 run_downstream_attention.py -m train -n wavlm_and_hubert_attention_1h -u hubert -d asr -o "config.downstream_expert.datarc.train=['1h']"
echo "\ndev:\n" > result.txt
python3 run_downstream_attention.py -m evaluate -e result/downstream/wavlm_and_hubert_attention_1h/dev-clean-best.ckpt -t 'dev-clean' | tail -n2 >> result.txt
echo "\n#################################\n" > result.txt
echo "\ntest:\n" > result.txt
python3 run_downstream_attention.py -m evaluate -e result/downstream/wavlm_and_hubert_attention_1h/dev-clean-best.ckpt -t 'test-clean' | tail -n2 >> result.txt

