# Hace que `common` sea un paquete y expone lo útil de forma cómoda
from .palette import PALETTE
from .vision  import (
    model,
    label_encoder,
    hands,
    mp_hands,
    mp_drawing,
    async_init_vision,
)