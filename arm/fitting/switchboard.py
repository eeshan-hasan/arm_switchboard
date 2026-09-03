from dataclasses import dataclass
import itertools
from tqdm import tqdm
from . import plots
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Switch:
    name: str
    n_variants : int
    variant: tuple[str, ...]
    human_readable: str
    default: 0

class Switchboard:
    def __init__(self,Switches,Variants=['attention_update_type','attention_update_dims'],ModelConfig=None):
        self.Switches = Switches
        self.Variants = Variants
        if(Variants is not None):
            self.make_matrix(Variants)
        self.ModelConfig=ModelConfig
            
    def make_matrix(self,Variants=['attention_update_type','attention_update_dims']):
        ''' Makes a matrix of all combiantions of switches and variants'''
        self.switch_list = []
        self.human_readable_list = []
        for switch in self.Switches:
            if(switch.name in Variants):
                self.switch_list.append(switch.variant)
                self.human_readable_list.append(switch.human_readable)
            else:
                self.switch_list.append([switch.variant[switch.default]])
        self.switch_matrix= list(itertools.product(*self.switch_list))
        self.human_readable_matrix= list(itertools.product(*self.human_readable_list))
        return self.switch_matrix
    
    def get_human_readable(self,index):
        human_readable_string = ""
        for switch_string in self.human_readable_matrix[index]:
            human_readable_string += switch_string + " | "
        return str(index)+": "+human_readable_string[:-3] 

    def __getitem__(self,index):
        self.make_matrix(self.Variants) 
        switch_states = self.get_state_index(index)
        return {
                    switch.name: switch_state
                    for switch_state, switch in zip(switch_states, self.Switches)
                }
        
    def get_state_index(self,index):
        return self.switch_matrix[index]
    
    def __str__():
        return self.switch_list.__str__()
        
    def __len__(self):
        return len(self.switch_matrix)
    
    def fit(self,data,index=None,n_runs=10):
        if(self.ModelConfig is None):
            raise ('Model Config is not set')
        else:
            ModelConfig = self.ModelConfig
        if(index is not None):
            init_params = self.__getitem__(index)
            ModelConfig = ModelConfig(model_init=init_params).fit(data=data,n_runs=n_runs)
            return ModelConfig
        else:
            Configs = []
            for i in tqdm(range(len(self))):
                init_params = self.__getitem__(i)
                ModelConfig_1 = ModelConfig(model_init=init_params)
                ModelConfig_1.fit(data=data,n_runs=n_runs)
                Configs.append(ModelConfig_1)
            return Configs
    
    def run(self,index,data,foldername = "./Switchboard/Test/",n_runs=10):
        if(self.ModelConfig is None):
            raise ('Model Config is not set')
        else:
            ModelConfig = self.ModelConfig
        init_params = self[index]
        ModelConfig = ModelConfig(model_init=init_params)
        
        estimated_params = ModelConfig.estimated_params
        ModelConfig = ModelConfig.fit(data=data,n_runs=n_runs)

        ModelConfig.make_model_data(data)
        ModelConfig.calculate_information_metrics()
        ModelConfig.save(index,foldername=foldername+str(index)+'/')
        ModelConfig.write_json(index,foldername=foldername+str(index)+'/')

        plots.make_full_plot(ModelConfig)
        plt.savefig(foldername+str(index)+'/full_plot.png',dpi=300)

        plots.make_full_plot_ci(ModelConfig)
        plt.savefig(foldername+str(index)+'/full_plot_ci.png',dpi=300)
        
        
        #fit
        #save
        #plot
        pass
