
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
sudo chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
export PATH="$HOME/miniconda3/bin:$PATH"


# Conda 환셩 생성 및 활성화
conda create --name myenv python=3.11 -y
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    problem=${file%.*}
    python "$file" < "../input/${problem}_input" > "../output/${problem}_output"
done

# mypy 테스트 실행 및 mypy_log.txt 저장
for file in *.py; do
    mypy "$file" >> ../mypy_log.txt
done

# conda.yml 파일 생성
conda env export > conda.yml

# 가상환경 비활성화
conda deactivate