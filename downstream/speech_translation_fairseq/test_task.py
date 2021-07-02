from argparse import Namespace
from .S3prl_SpeechToTextTask import S3prl_SpeechToTextTask

args = Namespace(**{
    'task': 'speech_to_text',
    'data': '/livingrooms/public/CoVoST2/cv-corpus-6.1-2020-12-11/en/',
    'config_yaml': 'config_st_en_de.yaml',
    'max_tokens' : 40000,
    'criterion': 'label_smoothed_cross_entropy',
    'label_smoothing': 0.1,
    'arch': 's2t_transformer_xs',
    'seed': 1,
})

task = S3prl_SpeechToTextTask.setup_task(args)
task.load_dataset('test_st_en_de')
batch_itr = task.get_batch_iterator(
    task.dataset('test_st_en_de'), max_tokens=args.max_tokens,
)

print(task.dataset('test_st_en_de')[0])
# itr = batch_itr.next_epoch_itr()
# for batch in itr:
#     print(batch)
#     break