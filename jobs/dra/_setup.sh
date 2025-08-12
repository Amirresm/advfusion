STORAGE_ROOT="/home/amirresm/projects/def-fard/amirresm"

MACHINE="DRA"

echo "Loading modules..."
module load StdEnv/2023 gcc cuda arrow python/3.13 scipy-stack
ENV_PATH=$SLURM_TMPDIR/env
if [ -f $ENV_PATH ]; then
	echo "Activating existing virtual environment in '$ENV_PATH'..."
	source "$ENV_PATH/bin/activate"
else
	echo "Creating virtual environment in '$ENV_PATH'..."
	virtualenv --no-download "$ENV_PATH"
	source "$ENV_PATH/bin/activate"
fi
echo "Working directory: $(pwd)"

echo "Upgrading pip..."
pip install --no-index --upgrade pip
echo "Installing dependencies..."
pip install --no-index -r "$PROJECT_ROOT/requirements.txt"
pip install "$PROJECT_ROOT/adapters-1.2.0-py3-none-any.whl"
