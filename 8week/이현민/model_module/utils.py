

import torch


def save_model(model, 
               target_dir,
               model_name):

    model_save_path = target_dir + '/' + model_name
    
    print("[INFO] Saving model to: {}".format(model_save_path))
    
    torch.save(obj=model.state_dict(),
               f=model_save_path)
