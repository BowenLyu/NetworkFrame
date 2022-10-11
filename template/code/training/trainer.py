import os
import torch
import utils.general as utils
import numpy as np
from datetime import datetime
import GPUtil
import omegaconf
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter



class Basetrainner():
    def __init__(self, args):
        #load
        # with open(args.conf_path) as f:
        #     conf = yaml.safe_load(f)
        conf = omegaconf.OmegaConf.load(args.conf_path)

        #arguments
        self.io = conf.dataio
        self.train = conf['train']
        self.network_conf = conf.network
        self.plot = conf.plot

        # # assign GPU
        # if args.gpu == "auto":
        #     deviceIDs = GPUtil.getAvailable(order='memory', limit=2, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
        #                             excludeUUID=[])
        #     gpu = deviceIDs[0]
        # else:
        #     gpu = args.gpu
        # self.GPU_INDEX = gpu
        # os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)
        

        #create export folder
        utils.mkdir_ifnotexists(os.path.join('../',self.io.exps_folder_name))
        self.expdir = os.path.join('../', self.io.exps_folder_name, self.io.expname)
        utils.mkdir_ifnotexists(self.expdir)

        #create timestamp
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        # #debug folder
        # log_dir = os.path.join(self.expdir, self.timestamp, 'log')
        # self.log_dir = log_dir
        # utils.mkdir_ifnotexists(log_dir)
        # utils.configure_logging(True,False,os.path.join(self.log_dir,'log.txt'))

        #create plot folder
        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        #create parameter folder
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        self.model_params_path = os.path.join(self.checkpoints_path,self.model_params_subdir)
        self.optimizer_params_path = os.path.join(self.checkpoints_path, self.optimizer_params_subdir)
        utils.mkdir_ifnotexists(self.checkpoints_path)
        utils.mkdir_ifnotexists(self.model_params_path)
        utils.mkdir_ifnotexists(self.optimizer_params_path)

        #tensorboard
        self.summaries_dir = os.path.join(self.expdir, self.timestamp, 'summaries')
        self.writer = SummaryWriter(self.summaries_dir)

        #network
        self.ds = 0
        self.dataloader = 0
        
        self.network = 0

        self.loss = 0

        #optimizer
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.parameters(),
                    "lr": self.train.lr,
                   # "betas": (0.9, 0.999),
                   # "eps": 1e-08,
                   # "weight_decay": self.weight_decay
                },
            ])
        
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=0.3)

        print("Initialization has finished!")
 

    def run(self):
        print("Begin to train!")

        total_steps = 0

        for epoch in tqdm(range(self.train['nepoch'])):
            if epoch == self.train['save_frequency']:
                print("save checkpoints!")
                torch.save(self.network.state_dict(),self.checkpoints_path)

            for step, (model_input, gt) in enumerate(self.dataloader):
                start_time = time.time()
            
               # output = model(input)
               
                train_loss = 0

                self.writer.add_scalar("total_train_loss", train_loss, total_steps)
                # losses = loss(output,_)
                self.optimizer.zero_grad()
                train_loss.backward()

                # if self.train['clip_grad']:
                #     torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.)

                self.optimizer.step()

                total_steps += 1

        
        print("Training finished!")
