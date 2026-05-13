from dataclasses import dataclass
import itertools

@dataclass(frozen=True)
class Switch:
    name: str
    n_variants : int
    variant: tuple[str, ...]
    human_readable: str
    default: 0

class Switchboard:
    def __init__(self,Switches,Variants=['attention_update_type','attention_update_dims']):
        self.Switches = Switches
        self.Variants = Variants
        if(Variants is not None):
            self.make_matrix(Variants)
            
    def make_matrix(self,Variants=['attention_update_type','attention_update_dims']):
        ''' Makes a matrix of all combiantions of switches and variants'''
        self.switch_list = []
        for switch in self.Switches:
            if(switch.name in Variants):
                self.switch_list.append(switch.variant)
            else:
                self.switch_list.append([switch.variant[switch.default]])
        self.switch_matrix= list(itertools.product(*self.switch_list))
        return self.switch_matrix

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
        
