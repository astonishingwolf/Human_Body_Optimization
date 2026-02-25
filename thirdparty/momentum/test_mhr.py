import torch
from mhr.mhr import MHR

mhr = MHR.from_files(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lod=1,
    )