from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """Base class for hint selection strategies"""
    
    @abstractmethod
    def run(self, dataset, output_path):
        """Run the strategy"""
        pass 