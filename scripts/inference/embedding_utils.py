import torch


def get_encoder_outputs(model, input_ids) -> torch.Tensor:
    """
    Get the encoder outputs for the given input_ids.
    Syntax for accessing encoder outputs differs based on whether
    the model exposes the encoder as an attribute (e.g. ByT5)
    or not (BART).
    """
    if hasattr(model, 'encoder'):
        with torch.no_grad():
            outputs = model.encoder(input_ids=input_ids)
        encoder_out = outputs.last_hidden_state.to('cpu')
    else:
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        encoder_out = outputs.encoder_last_hidden_state.to('cpu')
    del outputs
    return encoder_out