from random import shuffle 
import os
from glob import glob
import shutil
import re
import tqdm
from multiprocessing import Pool

from normalise import normalise

months = {'jan.': 'January', 'feb.': 'February', 'mar.': 'March', 'apr.': 'April', 'may': 'May', 'jun.': 'June', 'jul.': 'July', 'aug.': 'August', 'sep.': 'September', 'oct.': 'October', 'nov.': 'November', 'dec.': 'December', 'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April', 'jun': 'June', 'jul': 'July', 'aug': 'August', 'sep': 'September', 'oct': 'October', 'nov': 'November', 'dec': 'December'}

replace_words = {'&': 'and', '¡':'', 'r&b':'R and B', 'funtime':'fun time', 'español':'espanol', "'s":'s', 'palylist':'playlist'}
replace_vocab = {'ú':'u', 'ñ':'n', 'Ō':'O', 'â':'a'}

reservations = {'chyi':'chyi', 'Pre-Party':'pre party', 'Chu':'Chu', 'B&B':'B and B', '0944':'nine four four', 'Box':'Box', 'ain’t':'am not', 'Zon':'Zon', 'Yui':'Yui', 'neto':'neto', 'skepta':'skepta', '¡Fiesta':'Fiesta', 'Vue':'Vue', 'iheart':'iheart', 'disco':'disco'}
same = "klose la mejor música para tus fiestas dubstep dangles drejer listas".split(' ')
for word in same:
    reservations[word] = word

def word_normalise(words):
    ret = []
    for word in words:
        if word.lower() in months:
            word = months[word.lower()]
        if word.lower() in replace_words:
            word = replace_words[word.lower()]
        for regex in replace_vocab:
            word = re.sub(regex, '', word)
        #word = re.sub(r'(\S)([\.\,\!\?])', r'\1 \2', word)
        word = re.sub(r'[\.\,\!\?;\/]', '', word)
        ret.append(word)
    return ret

def sent_normalise(text, slots_split=None):
    norm_slots, norm_texts = [], []
    text_split = text.split(' ')
    if slots_split is None:
        slots_split = ['O']*len(text_split)
    for idx in range(len(text_split)):
        if text_split[idx] in '.,!?;/]':
            continue
        if text_split[idx] in reservations:
            for word in reservations[text_split[idx]].split(' '):
                norm_texts.append(word)
                norm_slots.append(slots_split[idx])
            continue
        norm_text = normalise(word_normalise([text_split[idx]]), variety="AmE", verbose=False)
        for phrase in norm_text:
            if phrase == '':
                continue
            for word in re.split(r' |\-', phrase):
                word = re.sub(r'[\.\,\!\?;\/]', '', word)
                if word == '':
                    continue
                norm_texts.append(word)
                norm_slots.append(slots_split[idx])
    return norm_slots, norm_texts


def process_raw_snips_file(file, out_f):
    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    with open(out_f, 'w') as f:
        for cnt, line in enumerate(content):
            text = line.split(' <=> ')[0]
            intent = line.split(' <=> ')[1]
            #[r.split(':')[0] if len(r.split(':')) == 2 else ' ' for r in x.split()]
            text_split = [x.replace('::', ':').split(':')[0] if len(x.replace('::', ':').split(':')) == 2 else ' ' for x in text.split()]
            text_entities = ' '.join(text_split)
            slots_split = [x.replace('::', ':').split(':')[1] for x in text.split()]
            slots_entities = ' '.join(slots_split)
            assert len(text_split) == len(slots_split), (text_split, slots_split)
            f.write('%d | BOS %s EOS | O %s | %s\n' % (cnt, text_entities, slots_entities, intent))

def remove_IBO_from_snipt_vocab_slot(in_f, out_f):
    with open(in_f) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    # get rid of BIO tag from the slots 
    for idx, line in enumerate(content):
        if line != 'O':
            content[idx] = line[len('B-'):]
    content = set(content) # remove repeating slots 

    with open(out_f, 'w') as f:
        for line in content:
            f.write('%s\n' % line)

def process_daniel_snips_file(content):
    content = [x.strip() for x in content]
    utt_ids = [x.split('\t', 1)[0] for x in content]
   
    valid_uttids = [x for x in utt_ids if x.split('-')[1] == 'valid']
    test_uttids  = [x for x in utt_ids if x.split('-')[1] == 'test']
    train_uttids = [x for x in utt_ids if x.split('-')[1] == 'train']

    utt2text, utt2slots, utt2intent = {}, {}, {}
    assert len(utt_ids) == len(set(utt_ids))

    # create utt2text, utt2slots, utt2intent
    for line in content:
        uttid, text, slots, intent = line.split('\t')
        if len(text.split()) != len(slots.split()): # detect 'empty' in text 
            assert len(text.split('  ')) == 2
            empty_idx = text.split().index(text.split('  ')[0].split()[-1]) + 1 
            slots_list = slots.split()
            del slots_list[empty_idx]
            cleaned_slots = ' '.join(slots_list)
            assert len(text.split()) == len(slots_list)
            cleaned_text = ' '.join(text.split())
            #print(cleaned_text, cleaned_slots)
        else: 
            (cleaned_text, cleaned_slots) = (text, slots)

        # get rid of the 'intent/' from all slot values 
        cleaned_slots = ' '.join([x.split('/')[1] if x != 'O' else x for x in cleaned_slots.split()])
        # strip the whitespaces before punctuations 
        #cleaned_text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', cleaned_text)
        
        utt2text[uttid]   = cleaned_text
        utt2slots[uttid]  = cleaned_slots
        utt2intent[uttid] = intent

    test_utt2text, test_utt2slots, test_utt2intent    = {}, {}, {}
    valid_utt2text, valid_utt2slots, valid_utt2intent = {}, {}, {}
    train_utt2text, train_utt2slots, train_utt2intent = {}, {}, {}
    for utt in valid_uttids:
         valid_utt2text[utt] = utt2text[utt]
         valid_utt2slots[utt] = utt2slots[utt]
         valid_utt2intent[utt] = utt2intent[utt]
    for utt in test_uttids:
         test_utt2text[utt] = utt2text[utt]
         test_utt2slots[utt] = utt2slots[utt]
         test_utt2intent[utt] = utt2intent[utt] 
    for utt in train_uttids:
         train_utt2text[utt] = utt2text[utt]
         train_utt2slots[utt] = utt2slots[utt]
         train_utt2intent[utt] = utt2intent[utt]
   
    assert len(set(valid_utt2intent.values())) == len(set(test_utt2intent.values())) == len(set(train_utt2intent.values())) == 7
    assert len(valid_utt2intent.keys()) == len(test_utt2intent.keys()) == 700
    assert len(train_utt2intent.keys()) == 13084
  
    def __return_set_of_slots(utt2slots):
        all_slots = []
        for slot in utt2slots.values():
            all_slots.extend(slot.split())
        unique_slots = set(all_slots)

        return unique_slots
    
    assert len(__return_set_of_slots(valid_utt2slots)) == len(__return_set_of_slots(test_utt2slots)) == \
        len(__return_set_of_slots(train_utt2slots)) == 40
  
    return (train_utt2text, train_utt2slots, train_utt2intent), \
           (valid_utt2text, valid_utt2slots, valid_utt2intent), \
           (test_utt2text, test_utt2slots, test_utt2intent)

def map_and_link_snips_audio(snips_audio_dir, link_dir):

    # traverse through snips_audio_dir
    result = [y for x in os.walk(snips_audio_dir) for y in glob(os.path.join(x[0], '*.mp3'))]

    for path in result: 
        person   = path.split('/')[8].split('_')[1]
        filename = path.split('/')[-1]
        if filename[:5] != 'snips':
            continue
        uttid = filename.split('.')[0]
        new_uttid = person + '-' + filename
        partition = uttid.split('-')[1]
        destination = os.path.join(link_dir, partition, new_uttid)
        shutil.copyfile(path, destination)

def create_multispk_for_snips(output_dir):
    speakers = "Aditi Amy Brian Emma Geraint Ivy Joanna Joey Justin Kendra Kimberly Matthew Nicole Raveena Russell Salli".split(' ')
    dataset_info = [{'split':'test', 'num_utts':700}, {'split':'valid', 'num_utts':700}, {'split':'train', 'num_utts':13084}]
    test_out_f = open(os.path.join(output_dir, 'all.iob.snips.txt'), 'w')
    for data in dataset_info:
        num_utts = data['num_utts']
        split = data['split']
        with open(os.path.join(output_dir, 'single-matched-snips.%s.w-intent'%split)) as f:
            content = f.readlines()
        utt2line = {x.strip().split()[0]:x.strip() for x in content}
        for spk in speakers:
            for num in range(num_utts):
                uttid  = "%s-snips-%s-%d"%(spk, split, num) #mp3.split('/')[-1].split('.')[0]
                line   = utt2line["snips-%s-%d"%(split, num)] #'-'.join(uttid.split('-')[1:])]
                text   = line.split('\t')[1].upper()
                slots  = line.split('\t')[2]
                intent = line.split('\t')[3] 
                test_out_f.write('%s BOS %s EOS\tO %s %s\n' % (uttid, text, slots, intent))
    test_out_f.close()

def apply_text_norm_and_modify_slots(all_tsv, output_dir):
    
    train_dirs, valid_dirs, test_dirs = process_daniel_snips_file(all_tsv)
    # test 
    test_file = open(os.path.join(output_dir, 'single-matched-snips.test.w-intent'), 'w')
    vocab_slot = {}
    for uttid in tqdm.tqdm(test_dirs[0].keys(), desc='Text Normalising on testing set'):
        text = test_dirs[0][uttid]
        slots = test_dirs[1][uttid]
        intent = test_dirs[2][uttid]
        slots_split = slots.split()
        for s in slots_split:
            vocab_slot.setdefault(s, 0)
            vocab_slot[s] += 1

        norm_slots, norm_texts = sent_normalise(text, slots_split)
        assert len(norm_texts) == len(norm_slots), (norm_texts, norm_slots)
     
        # write to file
        test_file.write('%s\t%s\t%s\t%s\n' % (uttid, ' '.join(norm_texts).upper(), ' '.join(norm_slots), intent))
    test_file.close()

    # valid 
    valid_file = open(os.path.join(output_dir, 'single-matched-snips.valid.w-intent'), 'w')
    for uttid in tqdm.tqdm(valid_dirs[0].keys(), desc='Text Normalising on validation set'):
        text = valid_dirs[0][uttid]
        slots = valid_dirs[1][uttid]
        intent = valid_dirs[2][uttid]
        slots_split = slots.split()
        for s in slots_split:
            vocab_slot.setdefault(s, 0)
            vocab_slot[s] += 1

        norm_slots, norm_texts = sent_normalise(text, slots_split)
        assert len(norm_texts) == len(norm_slots), (norm_texts, norm_slots)
     
        # write to file
        valid_file.write('%s\t%s\t%s\t%s\n' % (uttid, ' '.join(norm_texts).upper(), ' '.join(norm_slots), intent))
    valid_file.close()

    # train 
    train_file = open(os.path.join(output_dir, 'single-matched-snips.train.w-intent'), 'w')
    for uttid in tqdm.tqdm(train_dirs[0].keys(), desc='Text Normalising on training set'):
        text = train_dirs[0][uttid]
        slots = train_dirs[1][uttid]
        intent = train_dirs[2][uttid]
        slots_split = slots.split()
        for s in slots_split:
            vocab_slot.setdefault(s, 0)
            vocab_slot[s] += 1

        norm_slots, norm_texts = sent_normalise(text, slots_split)
        assert len(norm_texts) == len(norm_slots), (norm_texts, norm_slots)
     
        # write to file
        train_file.write('%s\t%s\t%s\t%s\n' % (uttid, ' '.join(norm_texts).upper(), ' '.join(norm_slots), intent))
    train_file.close()
    vocab_file = open(os.path.join(output_dir, 'slots.txt'), 'w')
    vocab_file.write('\n'.join(sorted(list(vocab_slot.keys()), key=lambda x:vocab_slot[x], reverse=True)))

def sox_func(inputs):
    files, root, out_root, speaker = inputs
    for name in tqdm.tqdm(files, desc='Process for speaker: '+speaker):
        if name.endswith(".mp3"):
            split = name.split('-')[1]
            out_dir = os.path.join(out_root, split)
            os.makedirs(out_dir, exist_ok=True)
            orig_file = os.path.join(root, name)
            new_file  = os.path.join(out_dir, speaker+'-'+name.split('/')[-1].split('.')[0] + '.wav')
            bashCommand = "sox " + orig_file + " -t wav -c 1 -r 16000 -b 16 -e signed-integer " + new_file
            r = os.popen(bashCommand).read()

def sox_mp3_to_wav(in_root, out_root):
    
    os.makedirs(out_root, exist_ok=True)
    pool = Pool(16)
    inputs = []
    for root, dirs, files in os.walk(in_root):
        print('[Processing] enter directory %s'%root)
        if not len(files):
            continue
        speaker = root.split('/')[-2].split('_')[1]
        print('[Processing] process %d audio files from speaker %s'%(len(files), speaker))
        inputs.append((files, root, out_root, speaker))
    pool.map(sox_func, inputs)

if __name__ == '__main__':

    import sys, os
    mode = sys.argv[1]
    if mode == 'text':
        repo_dir = sys.argv[2]
        dump_dir = sys.argv[3]
        os.makedirs(dump_dir, exist_ok=True)

        content = []
        content += open(os.path.join(repo_dir, 'data/nlu_annotation/valid')).readlines()[1:]
        content += open(os.path.join(repo_dir, 'data/nlu_annotation/test')).readlines()[1:]
        content += open(os.path.join(repo_dir, 'data/nlu_annotation/train')).readlines()[1:]
        apply_text_norm_and_modify_slots(content, dump_dir)
        create_multispk_for_snips(dump_dir)
    elif mode == 'audio':
        audio_dir = sys.argv[2]
        dump_dir = sys.argv[3]
        # Step: sox the snips *.mp3 to the correct format 
        sox_mp3_to_wav(audio_dir, dump_dir)
    else:
        print('Usage: python preprocess.py [text|audio] [data_path] [dump_path]')
