# AI Gender Bias Evaluation  
# AI性别偏见评估

本项目用于评估人工智能系统中的性别偏见，包括显性偏见和隐性偏见，并通过网络分析和结果整合，帮助研究者更好地理解和检测AI中的性别偏差现象。  
This project is designed to evaluate gender bias in artificial intelligence systems, including both explicit and implicit bias. Through network analysis and the integration of results, it helps researchers better understand and detect gender bias phenomena in AI.

---

## 项目结构 | Project Structure

- `01_explicit_bias_eval_flex.py`  
  显性性别偏见评估脚本，使用数据文件 `explicit_bias_data.xlsx` 进行分析。  
  Script for explicit gender bias evaluation using the dataset `explicit_bias_data.xlsx`.

- `02_implicit_bias_eval_flex.py`  
  隐性性别偏见评估脚本，使用数据文件 `implicit_bias_data.xlsx` 进行分析。  
  Script for implicit gender bias evaluation using the dataset `implicit_bias_data.xlsx`.

- `03_network_analysis_flex.py`  
  网络分析脚本，用于研究偏见在数据或模型中的传播和影响。  
  Script for network analysis to study the propagation and impact of bias in data or models.

- `04_run_all_flex.py`  
  一键运行所有评估流程的脚本，自动调用前述模块完成全流程分析。  
  Script for running all evaluation processes automatically.

- `05_integrate_all_outputs.py`  
  整合所有评估结果与分析输出，为后续报告生成或进一步挖掘提供支持。  
  Script for integrating outputs from all evaluations for further reporting or analysis.

- `explicit_bias_data.xlsx`  
  显性性别偏见评估用数据集。  
  Dataset for explicit gender bias evaluation.

- `implicit_bias_data.xlsx`  
  隐性性别偏见评估用数据集。  
  Dataset for implicit gender bias evaluation.

---

## 使用方法 | How to Use

1. 克隆仓库到本地：  
   Clone the repository:
   ```bash
   git clone https://github.com/Anniee001/ai-gender-bias-evaluation.git
   cd ai-gender-bias-evaluation
   ```

2. 安装依赖（如有）：  
   Install dependencies (if any):
   ```bash
   pip install -r requirements.txt
   ```

3. 按需运行各评估脚本：  
   Run evaluation scripts as needed:
   ```bash
   python 01_explicit_bias_eval_flex.py
   python 02_implicit_bias_eval_flex.py
   python 03_network_analysis_flex.py
   python 04_run_all_flex.py
   python 05_integrate_all_outputs.py
   ```

---

## 贡献方式 | Contribution

欢迎提交Issue与Pull Request以优化评估流程或补充相关数据与分析方法。  
Feel free to submit Issues or Pull Requests to improve the evaluation pipeline or add more data and analysis methods.

---

## License

本项目尚未指定License，请在使用前联系作者获取授权信息。  
This project is not yet licensed. Please contact the author for usage permission.

---

## 联系方式 | Contact

作者: [Anniee001](https://github.com/Anniee001)  
Author: [Anniee001](https://github.com/Anniee001)
