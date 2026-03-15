conda create -n visualsync python=3.10
conda activate visualsync
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

pip install --upgrade setuptools

cd Tracking-Anything-with-DEVA
pip install -e .
cd ..

cd Grounded-SAM-2
pip install -e .
pip install --no-build-isolation -e grounding_dino
pip install openai imageio dotenv transformers einops imgcat
pip install imageio[ffmpeg]
cd ..
