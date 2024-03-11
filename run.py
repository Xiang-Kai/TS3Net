import argparse
import os
import sys
import torch
import random
import numpy as np

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, default=1,
                    help='status')
parser.add_argument('--model_id', type=str, default='test',
                    help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# data loader
parser.add_argument('--data', type=str, default='ETTm1',
                    help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/',
                    help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv',
                    help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT',
                    help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                    help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=96, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false', default=True,
                    help='whether to use distilling in encoder, using this argument means not using distilling')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',
                    help='activation')
parser.add_argument('--output_attention', action='store_true',
                    help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision training')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

parser.add_argument('--d_model', type=int, default=768, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of ffn')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')

parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--gpt_layer', type=int, default=6)
parser.add_argument('--ln', type=int, default=0)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--weight', type=float, default=0)
parser.add_argument('--percent', type=int, default=5)

args = parser.parse_args()
print("")
print("")
print("")


if args.model == 'GPT4TS':
    print('Model Parameter Adaptation for GPT4TS')
else:
    print('Model Parameter Adaptation for Others')
    if (args.data == 'ETTh1') or ((args.data == 'ETTm1') and (args.pred_len in [336, 720])) or (
            (args.data == 'ETTm2') and (args.pred_len in [720])):
        args.d_model = 16
        args.d_ff = 32
    elif (args.data == 'ETTh2') or ((args.data == 'ETTm2') and (args.pred_len in [96, 192, 336])):
        args.d_model = 32
        args.d_ff = 32
    elif (args.data == 'ETTm1') and (args.pred_len in [96, 192]):
        args.d_model = 64
        args.d_ff = 64
    elif (args.data == 'Exchange') and (args.pred_len in [336, 720]):
        args.d_model = 32
        args.d_ff = 32
    elif (args.data == 'Exchange') and (args.pred_len in [96, 192]):
        args.d_model = 64
        args.d_ff = 64
    elif args.data == 'ECL':
        args.d_model = 256
        args.d_ff = 512
    elif args.data == 'Traffic':
        args.d_model = 512
        args.d_ff = 512
    elif args.data == 'ILI':
        args.d_model = 768
        args.d_ff = 768
    elif args.data == 'Weather':
        args.d_model = 32
        args.d_ff = 32
    elif args.data == 'M4':
        args.d_model = 32
        args.d_ff = 32
    else:
        print('Error-DataType !')

if args.task_name in ['long_term_forecast', 'imputation']:
    if args.data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        args.root_path = './dataset/ETT-small/'
        args.data_path = '{}.csv'.format(args.data)
        args.enc_in = 7
        args.dec_in = 7
        args.c_out = 7
    elif args.data == 'ECL':
        args.root_path = './dataset/electricity/'
        args.data_path = 'electricity.csv'
        args.enc_in = 321
        args.dec_in = 321
        args.c_out = 321
    elif args.data == 'Traffic':
        args.root_path = './dataset/traffic//'
        args.data_path = 'traffic.csv'
        args.enc_in = 862
        args.dec_in = 862
        args.c_out = 862
    elif args.data == 'ILI':
        args.root_path = './dataset/illness/'
        args.data_path = 'national_illness.csv'
        args.enc_in = 7
        args.dec_in = 7
        args.c_out = 7
    elif args.data == 'Exchange':
        args.root_path = './dataset/exchange_rate/'
        args.data_path = 'exchange_rate.csv'
        args.enc_in = 8
        args.dec_in = 8
        args.c_out = 8
    elif args.data == 'Weather':
        args.root_path = './dataset/weather/'
        args.data_path = 'weather.csv'
        args.enc_in = 21
        args.dec_in = 21
        args.c_out = 21
    else:
        print('Error-data !')

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
print('Args in experiment:')
print(args)

if args.task_name == 'long_term_forecast':
    Exp = Exp_Long_Term_Forecast
else:
    Exp = Exp_Long_Term_Forecast

if args.is_training:
    for ii in range(args.itr):
        setting = '{}_{}_{}_dm{}_df{}_el{}_fea{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.d_model,
            args.d_ff,
            args.e_layers,
            args.features,
        )
        path = os.path.join('./checkpoints/', setting)
        if not os.path.exists(path):
            os.makedirs(path)
        type = sys.getfilesystemencoding()
        sys.stdout = Logger(os.path.join('./checkpoints/' + setting, 'TerminalLog.txt'))

        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>start testing :  {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_pl{}_dm{}_df{}_sc{}_eb{}_nh{}_fea{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.patch_len,
        args.d_model,
        args.d_ff,
        args.stack_channel,
        args.e_blk,
        args.n_head,
        args.features,
    )
    exp = Exp(args)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)

    torch.cuda.empty_cache()
