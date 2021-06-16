import sys
sys.path.append('../')

from model import EQLM_Bert, EQLM_T5


checkpoint_path = "../ckpts/lr05_BDQLM_uncased_cleand_vocab_e9.ckpt"
MODEL_TYPE = 'bert-base-uncased'
output_path = "../ckpts/TILDE-BiQDL"

model = EQLM_Bert.load_from_checkpoint(model_type=MODEL_TYPE, checkpoint_path=checkpoint_path)
model.save(output_path)