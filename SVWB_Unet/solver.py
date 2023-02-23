import torch,os,random,time,rawpy,json

from tqdm import tqdm
from torch import optim
from torch import nn
from model import U_Net
from utils import *
from metrics import *
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

class Solver():
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # RAW args
        self.camera = config.camera
        self.raw = rawpy.imread("../"+self.camera+".dng")
        self.white_level = self.raw.white_level
        if self.camera == 'sony':
            self.white_level = self.white_level/4

        # Training config
        self.mode = config.mode
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.criterion = nn.MSELoss(reduction='mean')
        self.num_epochs_decay = config.num_epochs_decay
        self.save_epoch = config.save_epoch

        # Data loader
        self.data_root = config.data_root
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.input_type = config.input_type
        self.output_type = config.output_type

        # Models
        self.net = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.checkpoint = config.checkpoint

        # Visualize step
        self.save_result = config.save_result
        self.vis_step = config.vis_step
        self.val_step = config.val_step

        # Misc
        if config.checkpoint:
            self.train_date = self.checkpoint.split('/')[0] # get base directory from checkpoint
        else:
            self.train_date = time.strftime("%y%m%d_%H%M", time.localtime(time.time()))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize default path & SummaryWriter
        self.model_path = os.path.join(config.model_root,self.train_date)
        self.result_path = os.path.join(config.result_root,self.train_date+'_'+self.mode)
        if os.path.isdir(self.model_path) == False:
            os.makedirs(self.model_path)
        if os.path.isdir(self.result_path) == False and self.save_result == 'yes':
            os.makedirs(self.result_path)
        if self.mode == "train":
            self.log_path = os.path.join(config.log_root,self.train_date)
            self.writer = SummaryWriter(self.log_path)
            with open(os.path.join(self.model_path,'args.txt'), 'w') as f:
                json.dump(config.__dict__, f, indent=2)
            f.close()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        # build network, configure optimizer
        print("[Model]\tBuilding Network...")

        self.net = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)

        # Load model from checkpoint
        if self.checkpoint != None:
            ckpt_file = 'best.pt' if '/' not in self.checkpoint else os.path.split(self.checkpoint)[1]
            ckpt = os.path.join(self.model_path,ckpt_file)
            print("[Model]\tLoad model from checkpoint :", ckpt)
            self.net.load_state_dict(torch.load(ckpt))

        # multi-GPU
        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
        
        # GPU & optimizer
        self.net.to(self.device)
        self.optimizer = optim.Adam(list(self.net.parameters()),
                                      self.lr, [self.beta1, self.beta2])
        print("[Model]\tBuild Complete.")

    def train(self):
        print("[Train]\tStart training process.")
        best_net_score = 9876543210.
        best_mae_illum = 9876543210.
        best_psnr = 0.
        
        # Training
        for epoch in range(self.num_epochs):
            self.net.train(True)
            AE = 0
            trainbatch_len = len(self.train_loader)

            for i, batch in enumerate(self.train_loader):
                # prepare input
                if self.input_type == "rgb":
                    inputs = batch["input_rgb"].to(self.device)
                elif self.input_type == "uvl":
                    inputs = batch["input_uvl"].to(self.device)
                # prepare GT
                if self.output_type == "illumination":
                    GTs = batch["gt_illum"].to(self.device)
                elif self.output_type == "uv":
                    GTs = batch["gt_uv"].to(self.device)
                # prepare mask
                masks = batch["mask"].to(self.device)

                # inference
                pred = self.net(inputs)
                pred_detach = pred.detach()#.cpu()
                pred_loss = self.criterion(pred*masks, GTs*masks)

                # linear colorline loss
                if self.output_type == "illumination":
                    illum_map_rb = None
                    raise NotImplementedError
                elif self.output_type == "uv":
                    # difference of uv (inputs-pred) equals to illumination RB value
                    illum_map_rb = torch.exp((inputs[:,0:2,:,:]-pred)*masks)

                # Backprop & optimize network
                self.net.zero_grad()
                total_loss = pred_loss
                total_loss.backward()
                self.optimizer.step()

                # calculate pred_rgb & pred_illum & gt_illum
                input_rgb = batch["input_rgb"].to(self.device)
                gt_illum = batch["gt_illum"].to(self.device)
                gt_rgb = batch["gt_rgb"].to(self.device)
                if self.output_type == "illumination":
                    ones = torch.ones_like(pred_detach[:,:1,:,:])
                    pred_illum = torch.cat([pred_detach[:,:1,:,:],ones,pred_detach[:,1:,:,:]],dim=1)
                    pred_illum[:,1,:,:] = 1.
                    pred_rgb = apply_wb(input_rgb,pred_illum,pred_type='illumination')
                elif self.output_type == "uv":
                    pred_rgb = apply_wb(input_rgb,pred_detach,pred_type='uv')
                    pred_illum = input_rgb / (pred_rgb + 1e-8)
                    pred_illum[:,1,:,:] = 1.
                ones = torch.ones_like(gt_illum[:,:1,:,:])
                gt_illum = torch.cat([gt_illum[:,:1,:,:],ones,gt_illum[:,1:,:,:]],dim=1)

                # error metrics
                MAE_illum = get_MAE(pred_illum,gt_illum,tensor_type="illumination",mask=batch["mask"])
                MAE_rgb = get_MAE(pred_rgb,gt_rgb,tensor_type="rgb",camera=self.camera,mask=batch["mask"])
                PSNR = get_PSNR(pred_rgb,gt_rgb,white_level=self.white_level)

                # print training log & write on tensorboard & reset vriables
                print(f'[Train] Epoch [{epoch+1}/{self.num_epochs}] | ' \
                        f'Batch [{i+1}/{trainbatch_len}] | ' \
                        f'total_loss:{total_loss.item():.5f} | ' \
                        f'pred_loss:{pred_loss.item():.5f} | ' \
                        f'MAE_illum:{MAE_illum.item():.5f} | '\
                        f'MAE_rgb:{MAE_rgb.item():.5f} | '\
                        f'PSNR:{PSNR.item():.5f}')
                self.writer.add_scalar('train/total_loss',total_loss.item(),epoch * trainbatch_len + i)
                self.writer.add_scalar('train/pred_loss',pred_loss.item(),epoch * trainbatch_len + i)
                self.writer.add_scalar('train/MAE_illum',MAE_illum.item(),epoch * trainbatch_len + i)
                self.writer.add_scalar('train/MAE_rgb',MAE_rgb.item(),epoch * trainbatch_len + i)
                self.writer.add_scalar('train/PSNR',PSNR.item(),epoch * trainbatch_len + i)

            # lr decay
            if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                self.lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                print(f'Decay lr to {self.lr}')
            
            # Validation
            if epoch % self.val_step != 0:
                continue
            self.net.eval()

            valid_loss = 0
            valid_pred_loss = 0
            valid_MAE_illum = 0
            valid_MAE_rgb = 0
            valid_PSNR = 0
            valid_data_count = 0

            for i, batch in enumerate(self.valid_loader):
                # prepare input,GT,mask
                if self.input_type == "rgb":
                    inputs = batch["input_rgb"].to(self.device)
                elif self.input_type == "uvl":
                    inputs = batch["input_uvl"].to(self.device)
                if self.output_type == "illumination":
                    GTs = batch["gt_illum"].to(self.device)
                elif self.output_type == "uv":
                    GTs = batch["gt_uv"].to(self.device)
                masks = batch["mask"].to(self.device)

                # inference
                pred = self.net(inputs)
                pred_detach = pred.detach()
                pred_loss = self.criterion(pred, GTs)

                # linear colorline loss
                if self.output_type == "illumination":
                    illum_map_rb = None
                    raise NotImplementedError
                elif self.output_type == "uv":
                    # difference of uv (inputs-pred) equals to illumination RB value
                    illum_map_rb = torch.exp((inputs[:,0:2,:,:]-pred)*masks)
                total_loss = pred_loss
                
                # calculate pred_rgb & pred_illum & gt_illum
                input_rgb = batch["input_rgb"].to(self.device)
                gt_illum = batch["gt_illum"].to(self.device)
                gt_rgb = batch["gt_rgb"].to(self.device)
                if self.output_type == "illumination":
                    ones = torch.ones_like(pred_detach[:,:1,:,:])
                    pred_illum = torch.cat([pred_detach[:,:1,:,:],ones,pred_detach[:,1:,:,:]],dim=1)
                    pred_rgb = apply_wb(input_rgb,pred_illum,pred_type='illumination')
                elif self.output_type == "uv":
                    pred_rgb = apply_wb(input_rgb,pred_detach,pred_type='uv')
                    pred_illum = input_rgb / (pred_rgb + 1e-8)
                ones = torch.ones_like(gt_illum[:,:1,:,:])
                gt_illum = torch.cat([gt_illum[:,:1,:,:],ones,gt_illum[:,1:,:,:]],dim=1)
                pred_rgb = torch.clamp(pred_rgb,0,self.white_level)

                # error metrics
                MAE_illum = get_MAE(pred_illum,gt_illum,tensor_type="illumination",mask=batch["mask"])
                MAE_rgb = get_MAE(pred_rgb,gt_rgb,tensor_type="rgb",camera=self.camera,mask=batch["mask"])
                PSNR = get_PSNR(pred_rgb,gt_rgb,white_level=self.white_level)

                valid_loss += (total_loss.item() * len(inputs))
                valid_pred_loss += (pred_loss.item() * len(inputs))
                valid_MAE_illum += (MAE_illum.item() * len(inputs))
                valid_MAE_rgb += (MAE_rgb.item() * len(inputs))
                valid_PSNR += (PSNR.item() * len(inputs))
                valid_data_count += len(inputs)

            valid_loss /= valid_data_count
            valid_pred_loss /= valid_data_count
            valid_MAE_illum /= valid_data_count
            valid_MAE_rgb /= valid_data_count
            valid_PSNR /= valid_data_count

            # print validation log & write on tensorboard every epoch
            print(f'[Valid] Epoch [{epoch+1}/{self.num_epochs}] | ' \
                    f'total_loss: {valid_loss:.5f} | ' \
                    f'pred_loss: {valid_pred_loss:.5f} | ' \
                    f'MAE_illum: {valid_MAE_illum:.5f} | '\
                    f'MAE_rgb: {valid_MAE_rgb:.5f} | '\
                    f'PSNR: {valid_PSNR:.5f}')
            self.writer.add_scalar('validation/total_loss',valid_loss,epoch)
            self.writer.add_scalar('validation/pred_loss',valid_pred_loss,epoch)
            self.writer.add_scalar('validation/MAE_illum',valid_MAE_illum,epoch)
            self.writer.add_scalar('validation/MAE_rgb',valid_MAE_rgb,epoch)
            self.writer.add_scalar('validation/PSNR',valid_PSNR,epoch)
            
            # Save best U-Net model
            if valid_loss < best_net_score:
                best_net_score = valid_loss
                best_net = self.net.module.state_dict()
                print(f'Best net Score : {best_net_score:.4f}')
                torch.save(best_net, os.path.join(self.model_path, 'best_loss.pt'))
            if valid_MAE_illum < best_mae_illum:
                best_mae_illum = valid_MAE_illum
                best_net = self.net.module.state_dict()
                print(f'Best MAE_illum : {best_mae_illum:.4f}')
                torch.save(best_net, os.path.join(self.model_path, 'best_mae_illum.pt'))
            if valid_PSNR > best_psnr:
                best_psnr = valid_PSNR
                best_net = self.net.module.state_dict()
                print(f'Best PSNR : {best_psnr:.4f}')
                torch.save(best_net, os.path.join(self.model_path, 'best_psnr.pt'))
            
            # Save every N epoch
            if self.save_epoch > 0 and epoch % self.save_epoch == self.save_epoch-1:
                state_dict = self.net.module.state_dict()
                torch.save(state_dict, os.path.join(self.model_path, str(epoch)+'.pt'))

    def test(self):
        print("[Test]\tStart testing process.")
        self.net.eval()

        test_loss = []
        test_pred_loss = []
        test_MAE_illum = []
        test_MAE_rgb = []
        test_PSNR = []

        for i, batch in enumerate(self.test_loader):
            # prepare input,GT,mask
            if self.input_type == "rgb":
                inputs = batch["input_rgb"].to(self.device)
            elif self.input_type == "uvl":
                inputs = batch["input_uvl"].to(self.device)
            if self.output_type == "illumination":
                GTs = batch["gt_illum"].to(self.device)
            elif self.output_type == "uv":
                GTs = batch["gt_uv"].to(self.device)
            masks = batch["mask"].to(self.device)

            # inference
            pred = self.net(inputs)
            pred_detach = pred.detach()
            pred_loss = self.criterion(pred, GTs)

            # linear colorline loss
            if self.output_type == "illumination":
                illum_map_rb = None
                raise NotImplementedError
            elif self.output_type == "uv":
                # difference of uv (inputs-pred) equals to illumination RB value
                illum_map_rb = torch.exp((inputs[:,0:2,:,:]-pred))
            total_loss = pred_loss
            
            # calculate pred_rgb & pred_illum & gt_illum
            input_rgb = batch["input_rgb"].to(self.device)
            gt_illum = batch["gt_illum"].to(self.device)
            gt_rgb = batch["gt_rgb"].to(self.device)
            if self.output_type == "illumination":
                ones = torch.ones_like(pred_detach[:,:1,:,:])
                pred_illum = torch.cat([pred_detach[:,:1,:,:],ones,pred_detach[:,1:,:,:]],dim=1)
                pred_rgb = apply_wb(input_rgb,pred_illum,pred_type='illumination')
            elif self.output_type == "uv":
                pred_rgb = apply_wb(input_rgb,pred_detach,pred_type='uv')
                pred_illum = input_rgb / (pred_rgb + 1e-8)
            ones = torch.ones_like(gt_illum[:,:1,:,:])
            gt_illum = torch.cat([gt_illum[:,:1,:,:],ones,gt_illum[:,1:,:,:]],dim=1)
            pred_rgb = torch.clamp(pred_rgb,0,self.white_level)

            # input(pred_illum.shape)

            # error metrics
            MAE_illum = get_MAE(pred_illum,gt_illum,tensor_type="illumination",mask=batch["mask"])
            MAE_rgb = get_MAE(pred_rgb,gt_rgb,tensor_type="rgb",camera=self.camera,mask=batch["mask"])
            PSNR = get_PSNR(pred_rgb,gt_rgb,white_level=self.white_level)

            print(f'[Test] Batch [{i+1}/{len(self.test_loader)}] | ' \
                        f'total_loss:{total_loss.item():.5f} | ' \
                        f'pred_loss:{pred_loss.item():.5f} | ' \
                        f'MAE_illum:{MAE_illum.item():.5f} | '\
                        f'MAE_rgb:{MAE_rgb.item():.5f} | '\
                        f'PSNR:{PSNR.item():.5f}')

            test_loss.append(total_loss.item())
            test_pred_loss.append(pred_loss.item())
            test_MAE_illum.append(MAE_illum.item())
            test_MAE_rgb.append(MAE_rgb.item())
            test_PSNR.append(PSNR.item())

            if self.save_result == 'yes':
                # plot illumination map to R,B space
                plot_fig = plot_illum(pred_map=illum_map_rb.permute(0,2,3,1).reshape((-1,2)).cpu().detach().numpy(),
                                 gt_map=gt_illum[:,[0,2],:,:].permute(0,2,3,1).reshape((-1,2)).cpu().detach().numpy(),
                                 MAE_illum=MAE_illum,MAE_rgb=MAE_rgb,PSNR=PSNR)
                srgb_visualized = visualize(batch['input_rgb'][0],pred_rgb[0],batch['gt_rgb'][0],self.camera,concat=True)

                fname_base = batch["place"][0]+'_'+batch["illum_count"][0]

                cv2.imwrite(os.path.join(self.result_path,fname_base+'_plot.png'),plot_fig)
                cv2.imwrite(os.path.join(self.result_path,fname_base+'_vis.png'),cv2.cvtColor(srgb_visualized,cv2.COLOR_RGB2BGR))
                pred_illum_scale = pred_illum
                pred_illum_scale[:,1] *= 0.6
                save_image(fp=os.path.join(self.result_path,fname_base+'_illum.png'),tensor=pred_illum_scale[0].cpu().detach())

                pred_rgb_normalized = (pred_rgb[0] / self.white_level).cpu().detach()
                gamma_pred_rgb = torch.pow(pred_rgb_normalized,1/1.5)
                save_image(fp=os.path.join(self.result_path,fname_base+'_raw.png'),tensor=gamma_pred_rgb)

        print("loss :", np.nanmean(test_loss), np.median(test_loss), np.max(test_loss))
        print("pred_loss :", np.nanmean(test_pred_loss), np.median(test_pred_loss), np.max(test_pred_loss))
        print("MAE_illum :", np.nanmean(test_MAE_illum), np.median(test_MAE_illum), np.max(test_MAE_illum))
        print("MAE_rgb :", np.nanmean(test_MAE_rgb), np.median(test_MAE_rgb), np.max(test_MAE_rgb))
        print("PSNR :", np.nanmean(test_PSNR), np.median(test_PSNR), np.max(test_PSNR))