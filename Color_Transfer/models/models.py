
def create_model():
    from .colorhistogram_model import ColorHistogram_Model
    model = ColorHistogram_Model()        
    model.initialize()
    return model
