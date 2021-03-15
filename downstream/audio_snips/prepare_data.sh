if [ -s SNIPS/all.label.txt ];then
	echo 'Preprocessed text file exist, skip!'
else
	if [ ! -d aws-lex-noisy-spoken-language-understanding ];then
		echo 'Start downloading text files...'
		git clone https://github.com/aws-samples/aws-lex-noisy-spoken-language-understanding.git
	fi

	echo 'Start preparing text files...'
	mkdir SNIPS
	python text_preprocessing.py aws-lex-noisy-spoken-language-understanding SNIPS
	rm SNIPS/single*
fi

if [ -d SNIPS/valid ];then
	echo 'Preprocessed audio file exist, skip!'
else
	if [ ! -d audio_slu ];then
		echo 'Start downloading audio files...'
		wget https://shangwel-asr-evaluation.s3-us-west-2.amazonaws.com/audio_slu_v3.zip
		echo 'Start unzipping audio files...'
		unzip audio_slu_v3.zip > tmp.log
	fi

	echo 'Start converting audio files...'
	python audio_preprocessing.py audio_slu SNIPS
fi

