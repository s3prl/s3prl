git clone https://github.com/A2Zadeh/CMU-MultimodalSDK.git
cp convert_label.py CMU-MultimodalSDK
cd CMU-MultimodalSDK
wget http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd
wget https://raw.githubusercontent.com/A2Zadeh/CMU-MultimodalSDK/master/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei_std_folds.py
pip install h5py validators colorama tqdm requests pandas
python3 convert_label.py
mv CMU_MOSEI_Labels.csv ..
cd .. 
rm -rf CMU-MultimodalSDK