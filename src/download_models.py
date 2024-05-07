import gdown

# Download the files with specified names using gdown
gdown.download("https://drive.google.com/uc?id=1-9IVHrHrG907j4RtW78TL351QsVNWl9b", output="models/psum_twotasks.pth", quiet=False)
gdown.download("https://drive.google.com/uc?id=1-G8Dpi-XJ9tzANOh94s_OKc4boG5z6Qs", output="models/psum_twotasks_arabert.pth", quiet=False)
gdown.download("https://drive.google.com/uc?id=1BkE7qx9v7p_q3e98UzlVHkErxx8IzzaZ", output="models/arabertv02_LOGREG_MODEL.pkl", quiet=False)
gdown.download("https://drive.google.com/uc?id=1ppPTRFxA0mW23XJvGDmPRZXgvh3zugna", output="models/MARBERT_LOGREG_MODEL.pkl", quiet=False)
