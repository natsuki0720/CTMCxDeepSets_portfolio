from __future__ import annotations
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm, lognorm, gamma, weibull_min
from scipy.special import gamma as gamma_func  # ← 正しい gamma関数

import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import numpy as np

#ランダムに推移率行列を生成するクラス(上三角行列)
class TransitionRateMatrixGenerator:
    # コンストラクタ: 推移率行列の型を状態数により指定
    def __init__(self, states):
        self.num_states = states
        self.matrix = np.zeros((states, states))
        
    def generateMatrix(self, func):
        for i in range(self.num_states):
            func(i)  # 引数にインデックスを渡す
        return self.matrix
        
    # 推移率行列を一様分布により生成
    def setTransitionRateByUni(self, row_Index): 
        vector = np.zeros(self.num_states)
        for i in range(row_Index+1, self.num_states):
            vector[i] = np.random.uniform(0,1/self.num_states)
        sum = 0
        for j in range(0, len(vector)):
            sum += vector[j]
        vector[row_Index] = -sum
        self.matrix[row_Index] = vector
    
            
    # 対角成分をランダム生成し、それに合わせて各要素を決定
    def setTransitionRateFromDiagonal(self,row_Index):
        vector = np.zeros(self.num_states)
        for i in range(row_Index + 1, self.num_states):
            vector[i] = np.random.uniform()
        diag_val = 0
        if row_Index+1 != self.num_states:
            vector = self.__normalize(vector)
            diag_val = np.random.uniform(0,1)
             
        for i in range(row_Index + 1, self.num_states):
            vector[i] *= diag_val
        vector[row_Index] = -diag_val
        self.matrix[row_Index] = vector
    
    def __normalize(self, vector):
        sum = 0
        for i in vector:
            sum += i
        for i in range(0, len(vector)):
            vector[i] = vector[i]/sum
        return vector


class DiagonalTransitionRateMatrixGenerator:
    """
        対角成分のみの推移率行列を作成するクラス
    """
    def __init__(self,states):
        self.num_states = states
        self.matrix = np.zeros((self.num_states, self.num_states))
    
    def setDiagonalElement(self, row_index):
        if row_index < self.num_states-1:
            element = np.random.uniform(0.01,1)
            self.matrix[row_index][row_index] = -element
            self.matrix[row_index][row_index+1] = element
    
    def setDiagonalElement_byLifespan(self, row_index, lifespan):
        """
            寿命を一様分布に従うように対角成分を定義するメソッド
        """
        if row_index < self.num_states - 1:
            element = 1/np.random.uniform(1,lifespan)
            self.matrix[row_index][row_index] = -element
            self.matrix[row_index][row_index + 1] = element
    
    def generateMatrix(self, func,lifespan = 100):
        for i in range(0, self.num_states):
            func(i, lifespan)  
        return self.matrix
    
    
# 推移率行列から指定期間での推移確率行列を生成
class CalcProbmatrix:
    def __init__(self,matrix, delta_time):
        self.transitionRate_matrix = matrix
        self.delta_time = delta_time
        self.dim = len(matrix)
    # 水谷先生の論文中のAの行列を求めるメソッド
    def __calc_A(self, index):
        matrix  = np.eye(self.dim)
        for j in range(0,self.dim):
            if(j != index):
                mulMatrix = (self.transitionRate_matrix - self.transitionRate_matrix[j][j] * np.eye(self.dim))/(self.transitionRate_matrix[index][index] - self.transitionRate_matrix[j][j])
                matrix =  matrix @ mulMatrix 
        return matrix
    #推移確率を生成
    def calcProbmatrix(self):
        preb = np.zeros((self.dim,self.dim))
        for i in range(0, self.dim):
            A = self.__calc_A(i)
            mat = np.exp(self.delta_time * self.transitionRate_matrix[i][i]) * A
            preb += mat
        return preb

class gen_distributuion:
    def __init__(self,mixed_num = 3):
        self.mixed_num = mixed_num
        self.distribution_pool = ["gamma", "weibull_min", "lognorm"]
        self.components = []
        self.component_names = []
        self.mixed_ratio = self.set_mixed_rate()
    
    
    def set_mixed_rate(self,):
        alpha = np.ones(self.mixed_num)
        ratio = np.random.dirichlet(alpha)
        return ratio
    
    def sample_component(self, name):
        if name == 'gamma':
            shape = np.random.uniform(0.01,10)
            scale_max = 5/shape
            scale = np.random.uniform(0.1, scale_max)
            dist = gamma(a=shape, scale=scale)
            label = f"gamma(a={shape:.2f}, scale={scale:.2f})"
        
        elif name == 'weibull_min':
            M = 4.0  # 平均の上限
            c = np.random.uniform(0.5, 3.0)
            scale_max = M / gamma_func(1 + 1/c)
            scale = np.random.uniform(0.5, scale_max)

            dist = weibull_min(c=c, scale=scale)
            label = f"weibull(c={c:.2f}, scale={scale:.2f})"
        
        elif name == 'lognorm':
            mu = np.random.uniform(-1.0, np.log(4))  # 中央値 ≈ exp(μ) ∈ [1, 4]
            sigma_max = np.sqrt(2 * (np.log(4) - mu))
            sigma = np.random.uniform(0.2, sigma_max)  # 尾の厚さは制約付き

            dist = lognorm(s=sigma, scale=np.exp(mu))
            label = f"lognorm(μ={mu:.2f}, σ={sigma:.2f})"
        else:
            raise ValueError(f"Unknown distribution: {name}")
        
        return dist, label

    def set_components(self):
        self.components = []
        self.component_names = []
        for _ in range(self.mixed_num):
            name = np.random.choice(self.distribution_pool)
            dist, label = self.sample_component(name)
            self.components.append(dist)
            self.component_names.append(label)

    def pdf(self, x):
        total = np.zeros_like(x, dtype=float)
        for w, comp in zip(self.mixed_ratio, self.components):
            total += w * comp.pdf(x)
        return total
    
    def sample_one(self,):
    # 各サンプルに対して、どの分布から取るかを決定
        idx = np.random.choice(len(self.components), p=self.mixed_ratio)
        
        return self.components[idx].rvs()

        

class DataGenerator:
    def __init__(self, matrix, data_size, ):
        self.TR_M = matrix
        self.data_size = data_size
        self.num_states = len(matrix)
    
    def set_initialState_ratio(self):
        alpha = [1]*3
        ratio = np.random.dirichlet(alpha)
        return ratio
    
    # 初期劣化度の設定
    def set_initialState(self, ratio):
        p = np.random.rand()
        initial_state = None
        sum_ratio = 0
        for idx,r in enumerate(ratio):
            sum_ratio += r
            if p <= sum_ratio:
                initial_state = idx+1
                break
        if initial_state == None:
            initial_state = len(ratio)
        return initial_state
        
    # とりあえず一様分布
    def set_deltaTime(self, max):
        delta_time = np.random.uniform(0,max)
        return delta_time
    
    def set_deltaTime_log_normal(self):
        delta_time = np.random.lognormal(1, 0.5) 
        return delta_time
    
    class del_t_distribution():
        def __init__(self, mixed_num = 3):
            self.distribution = gen_distributuion(mixed_num)
            self.distribution.set_components()
        
        def get_del_t(self):
            return self.distribution.sample_one()
            
        
            
        
        
    
    def generate_sample(self, init_state, delta_t):
        probM = CalcProbmatrix(self.TR_M, delta_t).calcProbmatrix()
        probV = probM[init_state - 1]
        rand_int = np.random.rand()
        next_state = 0
        cumulative = 0
        for i in range(0, len(probV)):
            cumulative += probV[i]
            if(rand_int < cumulative):
                next_state = i+1
                break
        sample = np.array([init_state, next_state, delta_t])
        return sample
    
    def generate_dataFile(self, matrix,file_name,path):
        
        name = file_name +"_"+ str(self.data_size)+"_" + str(self.num_states)+".csv"
        # nested_directory = os.path.join("data","diagonal","nonFix")
        path = self.__setPath(name, path)
        matrix = np.array(matrix)
        np.savetxt(path, matrix,delimiter=",",fmt="%s",encoding='utf-8')
        
    def generate_matrix(self,func):
        data_matrix = []
        ratio = self.set_initialState_ratio()
        for _ in range(self.data_size):
            init_state = self.set_initialState(ratio)
            delta_t = round(func(),1) #一旦これ
            sampleVector = self.generate_sample(init_state, delta_t)
            data_matrix.append(sampleVector)
        data_matrix = np.array(data_matrix)
        data_matrix = self.__pad_matrix(data_matrix)
        matrix = np.vstack((self.TR_M,data_matrix))
        
        return matrix
    
   
    
        
    def generate_dataFile_fixTime(self, file_name, delta_t):
        data_matrix = []
        for i in range(0, self.data_size):
            init_state = self.set_initialState()
            sampleVector = self.generate_sample(init_state, delta_t)
            data_matrix.append(sampleVector)
        name = "fix"+ str(delta_t) +"_" + file_name + str(self.data_size)+"_" + str(self.num_states)+".csv"
        nested_directory = os.path.join("data","diagonal","fix"+str(delta_t))
        
        path = self.__setPath(name, nested_directory)
        
        data_matrix = np.array(data_matrix)
        data_matrix = self.__pad_matrix(data_matrix)
        matrix = np.vstack((self.TR_M,data_matrix))
        np.savetxt(path, matrix,delimiter=",",fmt="%s")
    
    def __setPath(self,file_name, directory = "data"):
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, file_name)
        return file_path
        
    def __pad_matrix(self, matrix):
        matrix = np.array(matrix)
        matrix_cols = matrix.shape[1]
        
        if matrix_cols < self.num_states:
            pad_cols = self.num_states - matrix_cols
            padding = np.zeros((matrix.shape[0],pad_cols))
            matrix = np.hstack((matrix,padding))
        return matrix
    

# 既存のクラス（このファイル内定義をそのまま利用）
# - DataGenerator
# - DiagonalTransitionRateMatrixGenerator

def _seed_for_index(base_seed: int, idx: int) -> int:
    ss = np.random.SeedSequence([base_seed, idx])
    return int(ss.generate_state(1)[0])

def _one_dataset_job(
    idx: int,
    out_dir: str,
    states: int,
    lifespan: float,
    min_n: int,
    max_n: int,
    base_seed: int,
) -> str:
    """
    元の逐次コード1回分をそのまま1ジョブとして並列実行する。
    - name は str(idx)
    - N（DataGenerator 第2引数）は [min_n, max_n] から乱択
    - del_t_distribution() → generate_matrix(distribution.get_del_t) を踏襲
    """
    # 乱数（np.random を使う実装に合わせて同期）
    child_seed = _seed_for_index(base_seed, idx)
    rng = np.random.default_rng(child_seed)
    np.random.seed(int(rng.integers(0, 2**31 - 1)))

    name = str(idx)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # あなたの元コードの手順に準拠
    TRMG = DiagonalTransitionRateMatrixGenerator(states)
    trm = TRMG.generateMatrix(TRMG.setDiagonalElement_byLifespan, lifespan)

    n_samples = int(rng.integers(min_n, max_n + 1))
    dg = DataGenerator(trm, n_samples)

    distribution = dg.del_t_distribution()
    M = dg.generate_matrix(distribution.get_del_t)

    # ファイル生成（実ファイル名の規約は DataGenerator に従う）
    dg.generate_dataFile(M, name, str(out))
    return str(out / name)

def _parse_args_parallel():
    p = argparse.ArgumentParser(description="Parallel runner compatible with the original per-iteration pattern.")
    p.add_argument("--count", type=int, required=True, help="反復回数（= 生成データセット数）")
    p.add_argument("--out-dir", type=str, required=True, help="出力ディレクトリ（例: /mnt/fast/datas/prototype3）")
    p.add_argument("--states", type=int, default=4, help="DiagonalTransitionRateMatrixGenerator の状態数")
    p.add_argument("--lifespan", type=float, default=100.0, help="setDiagonalElement_byLifespan に渡す値")
    p.add_argument("--min-n", type=int, default=5000, help="DataGenerator の最小サンプル数（第2引数）")
    p.add_argument("--max-n", type=int, default=5000, help="DataGenerator の最大サンプル数（第2引数）")
    p.add_argument("--workers", type=int, default=None, help="並列ワーカー数（未指定で論理コア数）")
    p.add_argument("--base-seed", type=int, default=20250912, help="再現用ベースシード")
    p.add_argument("--run-parallel", action="store_true", help="このフラグがあると並列実行ブロックを起動")
    return p.parse_args()

def _run_parallel_from_args(args):
    # 過剰並列（BLAS/OpenMPの多重スレッド）を回避
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    ctx = get_context("fork")  # UbuntuならforkでOK
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = [
            ex.submit(
                _one_dataset_job,
                i, args.out_dir, args.states, args.lifespan,
                args.min_n, args.max_n, args.base_seed
            )
            for i in range(args.count)
        ]
        for f in as_completed(futures):
            _ = f.result()  # 例外をここで表面化

# 既存の __main__ がなければこのブロックで起動できるようにする
if __name__ == "__main__":
    # 既存の単体実行用引数処理がある場合は、併存して問題ないように分岐
    args = _parse_args_parallel()
    if args.run_parallel:
        _run_parallel_from_args(args)
# ===== ここまで追記 =====

# ====== ここまで貼り付け ======