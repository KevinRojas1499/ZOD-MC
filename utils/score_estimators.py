import torch
import abc

from utils.densities import Distribution
import utils.optimizers as optimizers
import samplers.rejection_sampler as rejection_sampler
import samplers.ula as ula


class ScoreEstimator(abc.ABC):
    def __init__(self, dist: Distribution,
                 sde, device,def_num_batches=1,
                 def_num_samples=10000) -> None:
        self.sde = sde
        self.dist = dist
        self.device = device
        self.default_num_batches = def_num_batches
        self.default_num_samples = def_num_samples
        self.dim = self.dist.dim

    @abc.abstractmethod
    def score_estimator(self, x,tt, num_batches=None, num_rej_samples=None):
        pass
            
class ZODMC_ScoreEstimator(ScoreEstimator):
    
    def __init__(self, dist : Distribution, sde, device,
                 def_num_batches=1,
                 def_num_rej_samples=10000,
                 max_iters_opt=50
                 ) -> None:
        super().__init__(dist,sde,device,def_num_batches,def_num_rej_samples)
        
        # Set up distribution correctly
        dist.keep_minimizer = True
        minimizer = optimizers.newton_conjugate_gradient(torch.randn(dist.dim,device=device),
                                                         lambda x : -self.dist.log_prob(x), 
                                                         max_iters_opt)
        dist.log_prob(minimizer) # To make sure we update with the right minimizer
    
    def score_estimator(self, x,tt, num_batches=None, num_samples=None):
        scaling = self.sde.scaling(tt)
        variance_conv = (1/scaling)**2 - 1
        score_estimate = torch.zeros_like(x)
        num_batches = self.default_num_batches if num_batches is None else num_batches
        num_samples = self.default_num_samples if num_samples is None else num_samples

        assert num_batches > 0 and num_samples > 0, 'Number of samples needs to be a positive integer'
        
        mean_estimate = 0
        num_good_samples = torch.zeros((x.shape[0],1),device=self.device)
        for _ in range(num_batches):
            samples_from_p0t, acc_idx = rejection_sampler.get_samples(x/scaling, variance_conv,
                                                                                    self.dist,
                                                                                    num_samples, 
                                                                                    self.device)
            num_good_samples += torch.sum(acc_idx, dim=(1,2)).unsqueeze(-1).to(torch.double)/self.dim
            mean_estimate += torch.sum(samples_from_p0t * acc_idx,dim=1)
        num_good_samples[num_good_samples == 0] += 1 
        mean_estimate /= num_good_samples
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        return score_estimate

class RDMC_ScoreEstimator(ScoreEstimator):
    
    def __init__(self, dist : Distribution, sde, device,
                 def_num_batches=1,
                 def_num_samples=10000,
                 ula_step_size=0.01,
                 ula_steps=10,
                 initial_cond_normal=True) -> None:
        super().__init__(dist,sde,device,def_num_batches,def_num_samples)
        self.ula_step_size = ula_step_size
        self.ula_steps = ula_steps
        self.initial_cond_normal = initial_cond_normal
        
    def score_estimator(self, x,tt):
        scaling = self.sde.scaling(tt)
        inv_scaling = 1/scaling
        variance_conv = inv_scaling**2 - 1
        num_samples = self.default_num_samples
        score_estimate = torch.zeros_like(x)
        big_x = x.repeat_interleave(num_samples,dim=0)
        def grad_log_prob_0t(x0):
            return self.dist.grad_log_prob(x0) + scaling * (big_x - scaling * x0) / (1 - scaling ** 2)
        
        mean_estimate = 0
        x0 = big_x

        for _ in range(self.default_num_batches):
            if self.initial_cond_normal:
                x0 = inv_scaling * big_x + torch.randn_like(big_x) * variance_conv**.5
            samples_from_p0t = ula.get_ula_samples(x0,grad_log_prob_0t,self.ula_step_size,self.ula_steps)
            samples_from_p0t = samples_from_p0t.view((-1,num_samples, self.dim))
            
            mean_estimate += torch.sum(samples_from_p0t, dim = 1)
        mean_estimate/= (self.default_num_batches * self.default_num_samples)
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        return score_estimate

class RSDMC_ScoreEstimator(ScoreEstimator):
    
    def __init__(self, dist : Distribution, sde, device,
                 def_num_batches=1,
                 def_num_samples=10000,
                 ula_step_size=0.01,
                 ula_steps=10,
                 num_recursive_steps=3,
                 initial_cond_normal=True) -> None:
        super().__init__(dist,sde,device,def_num_batches,def_num_samples)
        self.ula_step_size = ula_step_size
        self.ula_steps = ula_steps
        self.initial_cond_normal = initial_cond_normal
        self.num_recursive_steps = num_recursive_steps
        
    def _recursive_langevin(self, x,tt,k=None):
        if k is None:
            k = self.num_recursive_steps
        if k == 0 or tt < .2:
            return self.dist.grad_log_prob(x)
        
        num_samples = self.default_num_samples
        scaling = self.sde.scaling(tt)
        # inv_scaling = 1/scaling
        h = self.ula_step_size      

        big_x = x.repeat_interleave(num_samples,dim=0) 
        x0 = big_x.detach().clone()   
        # x0 = inv_scaling * x0 + torch.randn_like(x0) * (inv_scaling**2 -1)  # q0 initialization
        for _ in range(self.ula_steps):
            score = self._recursive_langevin(x0, (k-1) * tt/k,k-1) + scaling * (big_x - scaling * x0)/(1-scaling**2)
            x0 = x0 + h * score + (2*h)**.5 * torch.randn_like(x0)
        x0 = x0.view((-1,num_samples,self.dim))
        mean_estimate = x0.mean(dim=1)
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        return score_estimate
    
    def score_estimator(self, x, tt):
        
        score_estimate = 0
        for _ in range(self.default_num_batches):
            score_estimate+= self._recursive_langevin(x,tt,self.num_recursive_steps)
        score_estimate/= self.default_num_batches
        return score_estimate
        

def get_score_function(config, dist : Distribution, sde, device):
    """
        The following method returns a method that approximates the score
    """
    grad_logdensity = dist.grad_log_prob
    dim = dist.dim

    
    def get_recursive_langevin(x,tt,k=config.num_recursive_steps):
        if k == 0 or tt < .2:
            return grad_logdensity(x)
        
        num_samples = config.num_estimator_samples
        scaling = sde.scaling(tt)
        # inv_scaling = 1/scaling
        h = config.ula_step_size      

        big_x = x.repeat_interleave(num_samples,dim=0) 
        x0 = big_x.detach().clone()   
        # x0 = inv_scaling * x0 + torch.randn_like(x0) * (inv_scaling**2 -1)  # q0 initialization
        for _ in range(config.num_sampler_iterations):
            score = get_recursive_langevin(x0, (k-1) * tt/k,k-1) + scaling * (big_x - scaling * x0)/(1-scaling**2)
            x0 = x0 + h * score + (2*h)**.5 * torch.randn_like(x0)
        x0 = x0.view((-1,num_samples,dim))
        mean_estimate = x0.mean(dim=1)
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        return score_estimate

        
    if config.score_method == 'p0t' and config.p0t_method == 'rejection':
        return ZODMC_ScoreEstimator(dist,sde,device,
                                    def_num_batches=config.num_estimator_batches,
                                    def_num_rej_samples=config.num_estimator_samples).score_estimator
    elif config.score_method == 'p0t' and config.p0t_method == 'ula':
        initial_cond_normal= True if config.rdmc_initial_condition.lower() == 'normal' else False
        return RDMC_ScoreEstimator(dist,sde,device,
                                def_num_batches=config.num_estimator_batches,
                                def_num_samples=config.num_estimator_samples,
                                ula_step_size=config.ula_step_size,
                                ula_steps=config.num_sampler_iterations,
                                initial_cond_normal=initial_cond_normal).score_estimator
    elif config.score_method == 'recursive':
        return RSDMC_ScoreEstimator(dist,sde,device,
                                def_num_batches=config.num_estimator_batches,
                                def_num_samples=config.num_estimator_samples,
                                ula_step_size=config.ula_step_size,
                                num_recursive_steps=config.num_recursive_steps,
                                ula_steps=config.num_sampler_iterations).score_estimator