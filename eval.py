
from model.evaluator import Evaluator
from model.dataloaders import Load_sfm_data
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", type=int, default=8,
                    help="The number of threads employed by the data loader")
parser.add_argument('--seed', type=int, default=0,
                    help='')
parser.add_argument('--GPUs', type=int, default=2,
                    help='The number of GPUs employed.')
parser.add_argument('--checkpoint', type=str, default="results/apt1_living_featloc++au.pth.tar",
                    help='The checkpoint_file')
parser.add_argument('--scene',type=str,default="apt1_living",
                    help="name of a scene in 7Scences or 12Scenes Dataset")
parser.add_argument('--version', type=int, default=2, choices=[0,1,2],
                    help='The version will be trained, 1-FeatLoc, 2-FeatLoc+, 3-FeatLoc++')
parser.add_argument('--is_plot', type=int, default=1,
                    help="visualization or not")

#--------------------------------------------------------------------------------------------------
args = parser.parse_args()
datadir_op = "dataset/Generated_Data/"+args.scene       
test_loader = Load_sfm_data(datadir_op, "test")
# model 
if args.version == 0:
    import model.FeatLoc as v0
    print("model version: FeatLoc")
    model = v0.MainModel()

elif args.version == 1:
    import model.FeatLocP as v1
    print("model version: FeatLoc+")
    model = v1.MainModel()
    
elif args.version == 2:
    import model.FeatLocPP as v2
    print("model version: FeatLoc++")
    model = v2.MainModel()
else:
    raise "Doesn't exist this model"


evaler = Evaluator(model , test_loader, args)
m,n = evaler.eval_sfm()




















