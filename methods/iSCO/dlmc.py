import torch
import random
import math

class DLMCSampler:
    def __init__(self):
        pass
    def get_init_sample(self,model):
        sample = torch.bernoulli(torch.full((model.num_instances,model.max_num_nodes,), 0.5)).to(model.device).detach()
        
        return sample
    
    def step(self,path_length,tensor_dict,temperature,model,sample,success_count):
        
        bsize = sample.shape[0]
        x = sample.requires_grad_()
        energy_x,grad_x = model.get_energy(tensor_dict,x)
        b_idx = torch.arange(bsize).to(x.device)
        prob_backward_list = []
        prob_forward_list = []

        # with torch.no_grad():
        cur_x = x.clone()
        idx_list = []
        for step in range(path_length):
            delta_x = -(2.0 * cur_x - 1.0)
            energy_cur_x,grad_cur_x = model.get_energy(tensor_dict,cur_x)
            score_change_x = -(delta_x * grad_cur_x) / (2 * temperature)
            nan_indices = torch.isnan(score_change_x)
            if torch.any(nan_indices):
                print("score_change_x 含有 NaN 值的索引：", nan_indices.nonzero())
            # score_change_x[torch.isinf(score_change_x)] = float(1e32)
            # score_change_x[torch.isnan(score_change_x)] = float(1e-32)
            score_change_x = score_change_x - torch.logsumexp(score_change_x, dim=-1, keepdim=True)
            prob_x_local = torch.softmax(score_change_x, dim=-1)
            if step > 0:
                prob_backward = torch.log(prob_x_local[:, index])
                prob_backward_list.append(prob_backward)
            index = torch.multinomial(prob_x_local, 1).view(-1)
            idx_list.append(index.view(-1, 1))
            cur_x[b_idx, index] = 1.0 - cur_x[b_idx, index]
            prob_forward = torch.log(prob_x_local[:, index])
            prob_forward_list.append(prob_forward)


        y = cur_x
        energy_y,grad_y = model.get_energy(tensor_dict,y)
        delta_y = -(2.0 * y - 1.0)
        score_change_y = -(delta_y * grad_y) / (2 * temperature)
        score_change_y = score_change_y - torch.logsumexp(score_change_y, dim=-1, keepdim=True)
        prob_y_local = torch.softmax(score_change_y, dim=-1)
        prob_backward = torch.log(prob_y_local[:, index])
        prob_backward_list.append(prob_backward)
        log_x2y = torch.sum(torch.cat(prob_forward_list))
        log_y2x = torch.sum(torch.cat(prob_backward_list))
        acc = torch.clamp(torch.exp(log_y2x - energy_y / ( temperature) - log_x2y +energy_x / ( temperature)),0,1)
        if torch.rand_like(acc) < acc:
            x = y
            energy_x = energy_y
            success_count += 1



        return x,energy_x,acc,success_count
    

def build_sampler():
    return DLMCSampler()





            

            
            