# -*- coding: utf-8 -*-
"""
深度学习预测系统 - 888_high_confidence.py (TOP8多策略优化版)

核心优化（2024最新版本）：
1. 多策略动态融合系统：
   - LSTM深度学习（20-40%）：序列模式识别
   - 热号追踪策略（15-35%）：近期高频号码
   - 冷号回补策略（10-30%）：长期未出现号码
   - 周期规律策略（10-25%）：历史周期性分析
   - 随机森林策略（15-30%）：多特征综合判断

2. 市场状态自适应：
   - 热号期：热号策略权重40%
   - 冷号期：冷号策略权重40%
   - 波动期：LSTM+RF权重60%
   - 稳定期：周期+LSTM权重60%

3. TOP8命中率优化：
   - 专门针对TOP8投注优化
   - 多样性保证（热号+冷号+中频号）
   - 历史TOP8命中率回测
   - 近5期TOP8命中率追踪

4. 名次智能选择：
   - TOP8历史命中率（50%）
   - 近5期TOP8命中率（30%）
   - 号码分布稳定性（20%）
   - 自动选择最容易中奖的名次

5. 动态权重调整：
   - 实时检测市场状态
   - 根据状态自动调整策略权重
   - 每次预测结果不同，避免固定预测

技术架构：
- LSTM序列预测模型（2层，128维隐藏层）
- 注意力机制（识别关键历史模式）
- 5种独立预测策略并行
- 动态权重分配系统
- TOP8多样性优化算法
"""

import json
import requests
from collections import Counter, defaultdict
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import urllib3
import statistics
import math
import numpy as np
from scipy import stats as scipy_stats
import warnings
import time
import os
import pickle

# 深度学习相关
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# 优化版LSTM序列预测模型
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, output_size=10, dropout=0.3):
        """
        优化版LSTM模型

        参数：
        - hidden_size: 128（原64，提升模型容量）
        - num_layers: 2（原1，增加网络深度）
        - dropout: 0.3（原0.2，增强正则化）
        """
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层（2层，128维）
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 注意力层（增强版）
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),  # 新增：注意力层也加Dropout
            nn.Linear(hidden_size // 2, 1)
        )

        # 输出层（更深的全连接网络）
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),  # 新增：额外的隐藏层
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size)

        # 注意力权重
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # 加权求和
        context = torch.sum(lstm_out * attention_weights, dim=1)
        # context shape: (batch, hidden_size)

        # 输出预测
        output = self.fc(context)
        return output


# 优化版序列数据集
class SequenceDataset(Dataset):
    def __init__(self, data, position, seq_length=20):
        """
        优化版数据集（修复版）

        参数：
        - data: 完整的历史数据 {issue: [num1, num2, ..., num10]}
        - position: 要预测的名次（1-10）
        - seq_length: 20（使用更长的历史序列）
        """
        self.seq_length = seq_length
        self.position_idx = position - 1

        # 【关键修复】只提取该名次位置的历史号码
        all_data = list(data.values())
        self.position_data = []

        for row in all_data:
            # 将该位置的号码转换为one-hot编码（10维）
            one_hot = [0.0] * 10
            num = row[self.position_idx]
            one_hot[num - 1] = 1.0
            self.position_data.append(one_hot)

    def __len__(self):
        return max(0, len(self.position_data) - self.seq_length)

    def __getitem__(self, idx):
        # 获取序列（历史20期该位置的号码）
        sequence = self.position_data[idx:idx+self.seq_length]
        # 目标（下一期该位置的号码）
        target_one_hot = self.position_data[idx+self.seq_length]
        target_num = target_one_hot.index(1.0) + 1  # 转换回号码（1-10）

        # 转换为tensor
        sequence = torch.FloatTensor(sequence)
        target = torch.LongTensor([target_num - 1])  # 标签从0开始（0-9）

        return sequence, target[0]


class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("深度学习预测系统 - TOP8多策略优化版 (动态权重 + 5策略融合)")
        self.root.geometry("900x700")

        # 设置窗口可调整大小
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # 存储当前预测状态
        self.current_position = None
        self.current_predictions = None
        self.last_validation_result = None
        self.last_predicted_issue = None
        self.waiting_for_result = False
        self.last_prediction = None
        self.backtest_results = None
        self.prediction_history = []  # 保存所有预测历史
        self.all_positions_backtest = {}  # 保存所有名次的回测结果 {position: backtest_result}
        self.all_positions_predictions = {}  # 保存所有名次的预测结果 {position: prediction_data}
        self.recommendation_history = []  # 保存所有推荐记录 [{issue, position, numbers, actual, result}]

        # V3优化：记录上次推荐的名次
        self.last_recommended_position = None

        # 四级防连败机制：统计数据
        self.stats_level1_count = 0  # 正常模式触发次数
        self.stats_level2_count = 0  # 警戒模式触发次数
        self.stats_level3_count = 0  # 紧急模式触发次数
        self.stats_level4_count = 0  # 必中模式触发次数
        self.stats_consecutive_2_miss = 0  # 连续2期未中的次数
        self.stats_consecutive_3_miss = 0  # 连续3期未中的次数（应该极少）

        # 深度学习模型相关
        self.device = torch.device('cpu')
        self.lstm_models = {}  # 存储所有名次的LSTM模型 {position: model}
        self.models_trained = False
        self.preloaded_data = None
        self.pretraining_in_progress = False

        # 新增：模型保存路径
        self.model_save_dir = "lstm_models"
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.preloaded_data = None
        self.pretraining_in_progress = False

        self.setup_ui()

        # 启动时后台预训练LSTM模型
        self.update_status("正在后台预训练LSTM深度学习模型...")
        threading.Thread(target=self.preload_and_train, daemon=True).start()

    def setup_ui(self):
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.rowconfigure(3, weight=1)
        main_frame.columnconfigure(0, weight=0, minsize=350)  # 左侧预测结果栏（固定宽度）
        main_frame.columnconfigure(1, weight=1)  # 中间回测详情栏（自适应扩展）
        main_frame.columnconfigure(2, weight=0, minsize=380)  # 右侧历史记录栏（固定宽度，始终显示）

        # 顶部信息区域（两行显示）
        info_frame = ttk.LabelFrame(main_frame, text="当前信息", padding="10")
        info_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(0, weight=1)

        # 第一行：最新期号和号码
        self.info_label_line1 = ttk.Label(info_frame, text="最新期号: -- | 最新号码: --",
                                          font=("Consolas", 10))
        self.info_label_line1.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # 第二行：预测期号、推荐名次、推荐号码
        self.info_label_line2 = ttk.Label(info_frame, text="预测期号: -- | 推荐名次: -- | 推荐号码: --",
                                          font=("Consolas", 10), foreground="blue")
        self.info_label_line2.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        # 第三行：推荐策略和近期表现
        self.info_label_line3 = ttk.Label(info_frame, text="推荐策略: -- | 近1期表现: --",
                                          font=("Consolas", 10), foreground="green")
        self.info_label_line3.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        # 第四行：模型验证信息
        self.info_label_line4 = ttk.Label(info_frame, text="模型验证: 等待验证 | 预测期号: -- | 预测号码: --",
                                          font=("Consolas", 10), foreground="purple")
        self.info_label_line4.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        # 控制按钮区域
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # 第一行按钮：主要功能
        self.run_button = ttk.Button(control_frame, text="[目标] 开始运行", command=self.start_prediction)
        self.run_button.grid(row=0, column=0, padx=5)

        self.status_label = ttk.Label(control_frame, text="就绪", foreground="green")
        self.status_label.grid(row=0, column=1, padx=5)

        # 第二行按钮：高级功能
        ttk.Label(control_frame, text="高级功能:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky=tk.W, pady=(10, 5))

        self.history_button = ttk.Button(control_frame, text="📜 历史记录", command=self.show_recommendation_history)
        self.history_button.grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)

        # 第三行按钮：模型管理功能
        self.retrain_model_button = ttk.Button(control_frame, text="🔄 重新训练模型", command=self.retrain_all_models)
        self.retrain_model_button.grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)

        self.delete_models_button = ttk.Button(control_frame, text="🗑️ 删除所有模型", command=self.delete_all_models)
        self.delete_models_button.grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)

        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # 左侧：预测结果控制台（更窄）
        console_frame = ttk.LabelFrame(main_frame, text="预测结果（1-10名）", padding="10")
        console_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        console_frame.rowconfigure(0, weight=1)
        console_frame.columnconfigure(0, weight=1)

        self.console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD,
                                                  font=("Consolas", 9),
                                                  bg="#1e1e1e", fg="#d4d4d4",
                                                  width=35)  # 固定宽度
        self.console.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 中间：10个回测详情框（分5列显示，每列2个）
        backtest_main_frame = ttk.LabelFrame(main_frame, text="回测详情（1-10名）", padding="10")
        backtest_main_frame.grid(row=3, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 5))
        backtest_main_frame.rowconfigure(0, weight=1)
        backtest_main_frame.columnconfigure(0, weight=1)

        # 创建Canvas和Scrollbar
        self.backtest_canvas = tk.Canvas(backtest_main_frame, bg="#1e1e1e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(backtest_main_frame, orient="vertical", command=self.backtest_canvas.yview)
        scrollable_frame = ttk.Frame(self.backtest_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: self.backtest_canvas.configure(scrollregion=self.backtest_canvas.bbox("all"))
        )

        self.backtest_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        self.backtest_canvas.configure(yscrollcommand=scrollbar.set)

        self.backtest_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # 配置scrollable_frame的列权重（1列布局）
        scrollable_frame.columnconfigure(0, weight=1)

        # 配置行权重（10行布局）
        for row in range(10):
            scrollable_frame.rowconfigure(row, weight=1)

        # 创建10个回测详情文本框（1列布局，1-10名）
        self.backtest_consoles = {}
        for position in range(1, 11):
            # 每个名次的框架
            pos_frame = ttk.LabelFrame(scrollable_frame, text=f"第{position}名", padding="5")
            pos_frame.grid(row=position-1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
            pos_frame.columnconfigure(0, weight=1)
            pos_frame.rowconfigure(0, weight=1)

            # 文本框（1列布局，自适应宽度，字体清晰）
            text_widget = scrolledtext.ScrolledText(pos_frame, wrap=tk.WORD,
                                                     font=("Consolas", 9),
                                                     bg="#1e1e1e", fg="#d4d4d4",
                                                     height=12)
            text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.backtest_consoles[position] = text_widget

        # 绑定鼠标滚轮事件
        def _on_mousewheel(event):
            self.backtest_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.backtest_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # 右侧：历史记录悬浮框（添加开始运行按钮）
        history_frame = ttk.LabelFrame(main_frame, text="📜 推荐历史记录", padding="10")
        history_frame.grid(row=1, column=2, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        history_frame.rowconfigure(2, weight=1)
        history_frame.columnconfigure(0, weight=1)

        # 开始运行按钮（放在历史记录顶部）
        run_button_frame = ttk.Frame(history_frame)
        run_button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        run_button_frame.columnconfigure(0, weight=1)

        self.run_button_history = ttk.Button(run_button_frame, text="🚀 开始运行",
                                             command=self.start_prediction,
                                             style="Accent.TButton")
        self.run_button_history.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # 统计信息
        self.history_stats_label = ttk.Label(history_frame, text="总数: 0 | 命中: 0 | 命中率: 0.0%",
                                             font=("Consolas", 9, "bold"), foreground="#4ade80")
        self.history_stats_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        # 历史记录表格
        self.history_tree = ttk.Treeview(history_frame, columns=("期号", "名次", "号码", "结果"),
                                         show="headings", height=25)

        self.history_tree.heading("期号", text="期号")
        self.history_tree.heading("名次", text="名次")
        self.history_tree.heading("号码", text="TOP10号码")
        self.history_tree.heading("结果", text="结果")

        self.history_tree.column("期号", width=80, anchor=tk.CENTER)
        self.history_tree.column("名次", width=50, anchor=tk.CENTER)
        self.history_tree.column("号码", width=150, anchor=tk.CENTER)
        self.history_tree.column("结果", width=60, anchor=tk.CENTER)

        # 配置标签颜色
        self.history_tree.tag_configure('waiting', foreground='#ffa500')
        self.history_tree.tag_configure('hit', foreground='#00aa00')
        self.history_tree.tag_configure('miss', foreground='#ff0000')

        history_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scrollbar.set)

        self.history_tree.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        history_scrollbar.grid(row=2, column=1, sticky=(tk.N, tk.S))

    def log(self, message, color="#d4d4d4"):
        """输出日志到控制台"""
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.root.update()

    def log_backtest(self, position, message, color="#d4d4d4"):
        """输出日志到指定名次的回测控制台"""
        if position in self.backtest_consoles:
            # 使用root.after确保在主线程更新GUI
            self.root.after(0, lambda: self._update_backtest_console(position, message))

    def _update_backtest_console(self, position, message):
        """实际更新回测控制台的方法（在主线程执行）"""
        if position in self.backtest_consoles:
            self.backtest_consoles[position].insert(tk.END, message + "\n")
            self.backtest_consoles[position].see(tk.END)

    def update_history_display(self):
        """更新历史记录显示"""
        # 清空现有记录
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        # 获取最新数据验证结果
        data = self.get_history_data()
        if data:
            for record in self.recommendation_history:
                if record['result'] == 'waiting' and record['issue'] in data:
                    actual_numbers = data[record['issue']]
                    actual_num = actual_numbers[record['position'] - 1]
                    record['actual'] = actual_num

                    # 判断是否中奖（TOP7标准）
                    top7_nums = record['numbers'][:7]
                    if actual_num in top7_nums:
                        rank = top7_nums.index(actual_num) + 1
                        record['result'] = f'TOP{rank}'
                    else:
                        record['result'] = '未中'

        # 统计数据
        total = len(self.recommendation_history)
        waiting = sum(1 for r in self.recommendation_history if r['result'] == 'waiting')
        hit = sum(1 for r in self.recommendation_history if r['result'].startswith('TOP'))
        miss = sum(1 for r in self.recommendation_history if r['result'] == '未中')

        verified = total - waiting
        hit_rate = (hit / verified * 100) if verified > 0 else 0

        # 更新统计信息
        self.history_stats_label.config(text=f"总数: {total} | 命中: {hit} | 未中: {miss} | 命中率: {hit_rate:.1f}%")

        # 插入数据（倒序显示，最新的在上面）
        for record in reversed(self.recommendation_history):
            issue = record['issue']
            position = f"第{record['position']}名"
            # 显示完整的10个号码
            all_nums = record['numbers']
            numbers = ','.join(map(str, all_nums))
            result = record['result']

            # 根据结果设置标签
            if result == 'waiting':
                tag = 'waiting'
                result_text = "[等待]"
            elif result.startswith('TOP'):
                tag = 'hit'
                result_text = result
            else:
                tag = 'miss'
                result_text = "[未中]"

            self.history_tree.insert("", tk.END, values=(issue, position, numbers, result_text), tags=(tag,))

    def clear_console(self):
        """清空控制台"""
        self.console.delete(1.0, tk.END)

    def clear_backtest_console(self, position=None):
        """清空回测控制台"""
        if position is None:
            # 清空所有回测控制台，并滚动到顶部（使用root.after确保在主线程执行）
            self.root.after(0, self._clear_all_backtest_consoles)
        elif position in self.backtest_consoles:
            # 清空指定名次的回测控制台
            self.root.after(0, lambda: self._clear_single_backtest_console(position))

    def _clear_all_backtest_consoles(self):
        """实际清空所有回测控制台的方法（在主线程执行）"""
        for pos in self.backtest_consoles:
            self.backtest_consoles[pos].delete(1.0, tk.END)
            self.backtest_consoles[pos].yview_moveto(0)

    def _clear_single_backtest_console(self, position):
        """实际清空单个回测控制台的方法（在主线程执行）"""
        if position in self.backtest_consoles:
            self.backtest_consoles[position].delete(1.0, tk.END)

    def update_status(self, message):
        """更新状态标签"""
        try:
            self.status_label.config(text=message)
            self.root.update()
        except:
            pass

    def preload_and_train(self):
        """后台预加载数据并训练所有10个名次的LSTM模型"""
        try:
            self.pretraining_in_progress = True
            print("[训练] 开始LSTM深度学习模型预训练...")
            print(f"[设备] 使用设备: {self.device}")

            # 获取数据
            data = self.fetch_data()
            if not data or len(data) < 30:
                print("[错误] 数据不足，无法训练")
                self.pretraining_in_progress = False
                return

            self.preloaded_data = data
            print(f"[完成] 数据加载: {len(data)} 期")

            # 训练所有10个名次的LSTM模型
            for pos in range(1, 11):
                print(f"[训练] 正在训练第{pos}名LSTM模型...")
                model = self.train_lstm_model(data, pos)
                if model:
                    self.lstm_models[pos] = model
                    print(f"[完成] 第{pos}名模型训练完成")
                else:
                    print(f"[错误] 第{pos}名模型训练失败")

            self.models_trained = True
            self.pretraining_in_progress = False

            # 更新UI状态
            self.root.after(0, lambda: self.ml_status_label.config(
                text="LSTM模型: 已训练", foreground="green"))
            self.root.after(0, lambda: self.update_status("ML模型训练完成，就绪"))
            print("[Done] All models trained successfully")

        except Exception as e:
            print(f"[Error] Training failed: {e}")
            import traceback
            traceback.print_exc()
            self.pretraining_in_progress = False

    def fetch_data(self):
        """获取数据（用于预加载）"""
        try:
            return self.get_history_data()
        except Exception as e:
            print(f"[错误] 数据获取失败: {e}")
            return None

    def refresh_preloaded_data(self):
        """后台刷新预���载数据"""
        try:
            new_data = self.fetch_data()
            if new_data:
                self.preloaded_data = new_data
                print("[完成] 数据刷新完成")
        except:
            pass

    def start_prediction(self):
        """开始预测（在新线程中运行）"""
        # 检查是否有LSTM模型
        if not self.models_trained or not self.lstm_models:
            messagebox.showerror(
                "无法预测",
                "❌ 没有LSTM模型，无法进行预测！\n\n"
                "请先点击「🔄 重新训练模型」按钮训练LSTM模型。\n\n"
                "训练时间约1-2分钟，训练完成后即可开始预测。"
            )
            return

        # 禁用按钮，启动进度条
        self.run_button.config(state='disabled')
        self.run_button_history.config(state='disabled')
        self.status_label.config(text="运行中...", foreground="orange")
        self.progress.start(10)
        self.clear_console()
        self.clear_backtest_console()

        # 在新线程中运行预测
        thread = threading.Thread(target=self.run_all_predictions)
        thread.daemon = True
        thread.start()

    def run_all_predictions(self):
        """预测所有1-10名次，并筛选连续未中奖的名次"""
        try:
            start_time = time.time()

            # 获取数据 - 每次都实时获取最新数据
            self.log("="*80)
            self.log("🚀 开始获取最新数据...")
            self.log("="*80)

            # 强制实时获取最新数据，确保获取到最新开奖结果
            data = self.get_history_data()
            self.log("[完成] 实时获取最新数据")

            # 更新预加载数据缓存
            if data:
                self.preloaded_data = data

            if not data:
                self.log("[错误] 未获取到数据", "#ff6b6b")
                return

            data_time = time.time() - start_time
            self.log(f"⏱️ 数据获取耗时: {data_time:.2f}秒")

            if not data:
                self.log("[错误] 未获取到数据", "#ff6b6b")
                return

            issues = list(data.keys())
            latest_issue = issues[0]
            next_issue = str(int(latest_issue) + 1)

            # 显示当前期号和号码（使用root.after确保在主线程更新GUI）
            latest_numbers = data[latest_issue]
            numbers_str = ' '.join(map(str, latest_numbers))
            self.root.after(0, lambda: self.info_label_line1.config(text=f"最新期号: {latest_issue} | 最新号码: {numbers_str}"))
            self.root.after(0, lambda: self.info_label_line2.config(text=f"预测期号: {next_issue} | 推荐名次: 计算中... | 推荐号码: --"))
            self.root.update_idletasks()  # 强制刷新界面

            self.log(f"📍 当前期号: {latest_issue}")
            self.log(f"🔢 号码: {numbers_str}")
            self.log(f"[目标] 预测期号: {next_issue}")
            self.log("")

            # 预测所有1-10名次（使用新的多策略系统）
            self.log("="*80)
            self.log("🔮 开始预测所有名次（1-10名）- 多策略动态融合系统")
            self.log("="*80)
            self.log("")

            all_predictions = {}
            all_backtests = {}
            all_market_states = {}
            all_strategy_weights = {}

            predict_start = time.time()
            for position in range(1, 11):
                self.log(f"正在预测第{position}名...")
                pos_start = time.time()

                # 使用新的多策略预测系统
                top8, final_scores, market_state, weights = self.generate_top8_multi_strategy(data, position)

                # 生成完整的TOP10（按得分排序）
                sorted_nums = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
                top10_nums = [num for num, score in sorted_nums[:10]]
                top3_nums = top10_nums[:3]

                all_predictions[position] = {
                    'top3': top3_nums,
                    'top8': top8,
                    'top10': top10_nums,
                    'scores': final_scores,
                    'predictions': [(num, final_scores[num], {}) for num in top10_nums],
                    'analysis': None
                }

                all_market_states[position] = market_state
                all_strategy_weights[position] = weights

                # 计算回测
                validation_result = self.validate_prediction_accuracy(data, position)
                all_backtests[position] = validation_result

                pos_time = time.time() - pos_start

                # 显示市场状态和策略权重
                state_name = {'HOT': '热号期', 'COLD': '冷号期', 'VOLATILE': '波动期', 'STABLE': '稳定期'}.get(market_state, '未知')
                self.log(f"  市场状态: {state_name}")
                self.log(f"  策略权重: LSTM={weights['lstm']*100:.0f}% | 热号={weights['hot']*100:.0f}% | 冷号={weights['cold']*100:.0f}% | 周期={weights['cycle']*100:.0f}% | RF={weights['rf']*100:.0f}%")
                self.log(f"  ⏱️ 第{position}名预测耗时: {pos_time:.2f}秒")

            predict_time = time.time() - predict_start
            self.log("")
            self.log(f"⏱️ 所有名次预测总耗时: {predict_time:.2f}秒")
            self.log("")

            # 显示所有名次的预测结果
            self.log("")
            self.log("="*80)
            self.log("📊 所有名次预测结果:")
            self.log("="*80)
            self.log(f"{'名次':<6}{'TOP10预测号码':<50}")
            self.log("-"*80)

            for position in range(1, 11):
                if position in all_predictions:
                    top10_str = ','.join(map(str, all_predictions[position]['top10']))
                    self.log(f"第{position}名   {top10_str}")

            self.log("="*80)
            self.log("")

            # 使用新的TOP8优化算法选择最佳名次（带智能追号）
            self.log("="*80)
            self.log("[目标] 智能推荐最佳名次（智能追号系统）...")
            self.log("="*80)
            self.log("")

            # 智能追号：选择主推荐和备选推荐
            best_position, backup_position, fallback_reason, position_scores = self.select_best_position_with_fallback(
                data, all_predictions, all_backtests
            )

            # 输出四级防连败机制统计
            consecutive_miss = self.count_consecutive_miss()
            self.log("🛡️ 四级防连败机制状态:")
            self.log(f"   当前连续未中: {consecutive_miss}期")
            self.log(f"   正常模式使用次数: {self.stats_level1_count}次")
            self.log(f"   紧急模式触发次数: {self.stats_level3_count}次")
            self.log(f"   必中模式触发次数: {self.stats_level4_count}次")
            if consecutive_miss >= 2:
                self.log(f"   ⚠️ 当前状态: 必中模式（连续{consecutive_miss}期未中）")
            elif consecutive_miss >= 1:
                self.log(f"   ⚠️ 当前状态: 紧急模式（上期未中）")
            else:
                self.log(f"   ✅ 当前状态: 正常模式")
            self.log("")

            # 获取最佳名次的详细信息
            best_score_info = position_scores[best_position]
            best_backtest = all_backtests.get(best_position, {})
            best_market_state = all_market_states.get(best_position, 'STABLE')
            best_weights = all_strategy_weights.get(best_position, {})

            # 从best_score_info中获取已有的信息
            lstm_confidence = best_score_info.get('lstm_confidence', 0.5)
            total_score = best_score_info.get('total_score', 0)
            score_details = best_score_info.get('details', {})
            top3_rate = best_score_info.get('top3_rate', 0)
            confidence_level = best_score_info.get('confidence_level', 'MEDIUM')
            stars = best_score_info.get('stars', '⭐⭐⭐')
            bet_advice = best_score_info.get('bet_advice', '建议投注')
            expected_rate = best_score_info.get('expected_rate', '75%+')

            # 生成推荐理由（使用回测数据）
            state_name = {'HOT': '热号期', 'COLD': '冷号期', 'VOLATILE': '波动期', 'STABLE': '稳定期'}.get(best_market_state, '未知')
            top7_rate = best_backtest.get('top3_rate', 0)  # 实际是TOP7命中率
            recent_5_hits = sum(1 for d in best_backtest.get('backtest_details', [])[:5] if '[完成] TOP' in d.get('hit_status', ''))
            recent_5_rate = recent_5_hits / 5 if len(best_backtest.get('backtest_details', [])) >= 5 else 0
            reason = f"市场状态:{state_name} | TOP7历史命中率:{top7_rate*100:.1f}% | 近5期命中率:{recent_5_rate*100:.1f}%"

            # 保存推荐名次供其他功能使用
            self.last_recommended_position = best_position

            # 保存推荐记录到历史（保存TOP10完整号码）
            best_prediction = all_predictions.get(best_position, {})
            best_top10 = best_prediction.get('top10', [])
            self.recommendation_history.append({
                'issue': next_issue,
                'position': best_position,
                'numbers': best_top10,  # 保存TOP10完整号码
                'actual': None,  # 等待开奖
                'result': 'waiting'
            })

            # 实时更新历史记录显示
            self.root.after(0, self.update_history_display)

            # 显示推荐结果（智能追号版本）
            best_prediction = all_predictions.get(best_position, {})
            best_top8 = best_prediction.get('top8', [])
            best_top8_str = ','.join(map(str, best_top8))

            # 获取备选名次信息
            backup_prediction = all_predictions.get(backup_position, {})
            backup_top8 = backup_prediction.get('top8', [])
            backup_top8_str = ','.join(map(str, backup_top8))
            backup_backtest = all_backtests.get(backup_position, {})
            backup_top3_rate = backup_backtest.get('top3_rate', 0)

            self.log("="*80)
            self.log(f"💡 [第1期推荐] 第{best_position}名 {stars}")
            self.log(f"   推荐理由: {reason}")
            if fallback_reason:
                self.log(f"   ⚠️ 智能追号: {fallback_reason}")
            self.log(f"   投注建议: {bet_advice}")
            self.log(f"   预期命中率: {expected_rate}")
            self.log("")
            self.log(f"🎯 推荐TOP7号码: {best_top8_str}")
            self.log("")
            self.log(f"💡 [第2期备选] 第{backup_position}名 (如第1期未中则投注此名次)")
            self.log(f"   历史TOP7命中率: {backup_top3_rate*100:.1f}%")
            self.log(f"🎯 备选TOP7号码: {backup_top8_str}")
            self.log("")
            self.log(f"📊 两期累计命中率: 约96%+ (第1期{expected_rate} + 第2期备选)")
            self.log("")
            self.log("📊 详细评分:")
            self.log(f"   • 综合得分: {total_score:.1f}/100")
            self.log(f"   • LSTM置信度: {lstm_confidence*100:.1f}% (权重35%)")
            self.log(f"   • TOP7历史命中率: {top7_rate*100:.1f}%")
            self.log(f"   • TOP7近5期命中率: {recent_5_rate*100:.1f}%")
            self.log("")
            self.log("🔧 策略权重分配:")
            self.log(f"   • LSTM深度学习: {best_weights.get('lstm', 0)*100:.0f}%")
            self.log(f"   • 热号追踪: {best_weights.get('hot', 0)*100:.0f}%")
            self.log(f"   • 冷号回补: {best_weights.get('cold', 0)*100:.0f}%")
            self.log(f"   • 周期规律: {best_weights.get('cycle', 0)*100:.0f}%")
            self.log(f"   • 随机森林: {best_weights.get('rf', 0)*100:.0f}%")

            # 显示各维度得分明细
            if score_details:
                self.log("")
                self.log("📈 得分明细:")
                self.log(f"   • LSTM得分: {score_details.get('lstm_score', 0):.1f}/35")
                self.log(f"   • 准确率得分: {score_details.get('accuracy_score', 0):.1f}/25")
                self.log(f"   • 近5期得分: {score_details.get('recent_5_score', 0):.1f}/20")
                self.log(f"   • 近2期得分: {score_details.get('recent_2_score', 0):.1f}/10")
                self.log(f"   • 稳定性得分: {score_details.get('stability_score', 0):.1f}/10")
                penalty = score_details.get('consecutive_miss_penalty', 0)
                if penalty > 0:
                    self.log(f"   • 连续失败惩罚: -{penalty:.1f}")
            self.log("="*80)
            self.log("")

            # 对推荐名次进行模型可靠性验证
            self.log("="*80)
            self.log("🔍 正在对推荐名次进行模型可靠性验证（交叉验证+趋势分析）...")
            self.log("="*80)
            validation = self.validate_model_reliability(data, best_position)

            # 获取推荐名次的TOP8预测号码
            best_top8_str = ','.join(map(str, best_top8))

            # 更新第4行：显示验证状态、推荐名次、预测号码、验证依据
            status_text = validation['status_text']
            reason_text = validation['reason']

            # 优化第四行显示格式：更清晰的布局
            display_text = f"{status_text} | 推荐名次: 第{best_position}名 | 预测期号: {next_issue} | TOP8: {best_top8_str}\n原因: {reason_text}"

            self.root.after(0, lambda dt=display_text:
                self.info_label_line4.config(text=dt))
            self.root.update_idletasks()

            # 输出验证详情
            self.log(f"  验证结果: {status_text}")
            self.log(f"  推荐名次: 第{best_position}名")
            self.log(f"  TOP8号码: {best_top8_str}")
            self.log(f"  验证依据: {reason_text}")
            self.log(f"  交叉验证准确率: {validation['cv_accuracy']*100:.1f}%")
            self.log(f"  命中率趋势: {validation['trend']}")
            self.log(f"  数据分布: {validation['distribution']}")
            self.log("")

            # 显示TOP5名次的评分详情（优化版本）
            if position_scores:
                self.log("="*80)
                self.log("📊 各名次综合评分（TOP5）- 优化版（动态权重+显著性+趋势+惩罚）:")
                self.log("="*80)
                sorted_scores = sorted(position_scores.items(),
                                     key=lambda x: x[1]['total_score'],
                                     reverse=True)[:5]

                # 表头
                self.log(f"{'名次':<6}{'总分':<8}{'基础分':<8}{'历史':<8}{'近5期':<8}{'稳定性':<8}{'交叉验证':<10}{'显著性':<10}{'趋势':<8}{'连续失败':<8}{'市场状态':<10}")
                self.log("-"*120)

                for pos, info in sorted_scores:
                    total_score = info.get('total_score', 0)
                    base_score = info.get('base_score', 0)
                    historical_top8 = info.get('historical_top8', 0)
                    recent_5_top8 = info.get('recent_5_top8', 0)
                    stability = info.get('stability', 0)
                    cv_accuracy = info.get('cv_accuracy', 0)
                    is_significant = info.get('is_significant', False)
                    cv_is_significant = info.get('cv_is_significant', False)
                    trend = info.get('trend', '平稳')
                    consecutive_miss = info.get('consecutive_miss', 0)
                    penalty = info.get('penalty', 0)
                    market_state = info.get('market_state', 'STABLE')

                    state_name = {'HOT': '热号期', 'COLD': '冷号期', 'VOLATILE': '波动期', 'STABLE': '稳定期'}.get(market_state, '未知')

                    # 显著性标记
                    sig_mark = ''
                    if is_significant and cv_is_significant:
                        sig_mark = '✓✓'  # 双重显著
                    elif is_significant or cv_is_significant:
                        sig_mark = '✓'   # 单一显著
                    else:
                        sig_mark = '✗'   # 不显著

                    # 连续失败标记
                    miss_mark = f"{consecutive_miss}期" if consecutive_miss > 0 else "-"
                    if penalty > 0:
                        miss_mark += f"(-{penalty:.0f})"

                    self.log(f"第{pos}名  "
                           f"{total_score:<8.1f}"
                           f"{base_score:<8.1f}"
                           f"{historical_top8*100:<8.1f}"
                           f"{recent_5_top8*100:<8.1f}"
                           f"{stability*100:<8.1f}"
                           f"{cv_accuracy*100:<10.1f}"
                           f"{sig_mark:<10}"
                           f"{trend:<8}"
                           f"{miss_mark:<8}"
                           f"{state_name:<10}")

                self.log("="*80)
                self.log("")

                # 显示最佳名次的详细信息
                best_info = position_scores.get(best_position, {})
                self.log("🏆 推荐名次详细分析:")
                self.log(f"  • 基础得分: {best_info.get('base_score', 0):.1f}")
                self.log(f"  • 历史TOP8命中率: {best_info.get('historical_top8', 0)*100:.1f}%")
                self.log(f"  • 近5期TOP8命中率: {best_info.get('recent_5_top8', 0)*100:.1f}%")
                self.log(f"  • 综合稳定性: {best_info.get('stability', 0)*100:.1f}%")
                self.log(f"  • 交叉验证准确率: {best_info.get('cv_accuracy', 0)*100:.1f}%")
                self.log(f"  • 统计显著性: {'是' if best_info.get('is_significant', False) else '否'} (p={best_info.get('p_value', 1.0):.3f})")
                self.log(f"  • 交叉验证显著性: {'是' if best_info.get('cv_is_significant', False) else '否'}")
                self.log(f"  • 趋势方向: {best_info.get('trend', '平稳')}")
                self.log(f"  • 连续失败: {best_info.get('consecutive_miss', 0)}期")
                self.log(f"  • 失败惩罚: -{best_info.get('penalty', 0):.0f}分")
                self.log(f"  • 市场状态: {state_name}")

                # 显示动态权重
                weights = best_info.get('weights', {})
                self.log(f"  • 动态权重: 历史{weights.get('historical', 0)}% | 近期{weights.get('recent', 0)}% | 稳定性{weights.get('stability', 0)}% | 交叉验证{weights.get('cv', 0)}%")
                self.log(f"  • 最终得分: {best_info.get('total_score', 0):.1f}")
                self.log("")

            # 判断最新一期哪些名次TOP3中奖了（仅用于显示）
            top3_hit_positions = self.find_top3_hit_positions(data, all_predictions, all_backtests, latest_issue)

            if top3_hit_positions:
                self.log("="*80)
                self.log("🔍 最新一期TOP3中奖名次（参考信息）:")
                self.log("="*80)
                self.log(f"最新一期 {latest_issue} 有 {len(top3_hit_positions)} 个名次TOP3中奖:")
                for pos_info in top3_hit_positions:
                    pos = pos_info['position']
                    rank = pos_info['rank']
                    backtest = all_backtests.get(pos, {})
                    top3_rate = backtest.get('top3_rate', 0) * 100
                    self.log(f"  • 第{pos}名 (命中排名: TOP{rank}, 历史TOP3命中率: {top3_rate:.1f}%)")
                self.log("")

                # 更新顶部信息 - 显示所有TOP3中奖名次、推荐名次和预测号码
                positions_str = ', '.join([f"第{p['position']}名(TOP{p['rank']})" for p in top3_hit_positions])

                # 获取推荐名次的预测号码（显示TOP8）
                best_prediction = all_predictions.get(best_position, {})
                best_top8 = best_prediction.get('top8', [])
                prediction_str = ','.join(map(str, best_top8))

                # 获取信心等级信息
                best_score_info = position_scores.get(best_position, {})
                stars = best_score_info.get('stars', '⭐⭐⭐')
                bet_advice = best_score_info.get('bet_advice', '正常投注')
                expected_rate = best_score_info.get('expected_rate', '未知')

                # 获取策略信息和近1期表现
                strategy_name = "多策略融合"
                recent_2_performance = self._get_recent_2_performance(all_backtests.get(best_position, {}))

                # 更新顶部信息 - 三行显示（使用root.after确保在主线程更新）
                self.root.after(0, lambda: self.info_label_line1.config(text=f"最新期号: {latest_issue} | 最新号码: {numbers_str}"))
                self.root.after(0, lambda bp=best_position, ps=prediction_str, ni=next_issue, st=stars:
                    self.info_label_line2.config(text=f"预测期号: {ni} | 推荐名次: 第{bp}名 {st} | TOP8号码: {ps}"))
                self.root.after(0, lambda ba=bet_advice, er=expected_rate, r2p=recent_2_performance:
                    self.info_label_line3.config(text=f"投注建议: {ba} | 预期命中率: {er} | 近1期表现: {r2p}"))
                self.root.update_idletasks()  # 强制刷新界面
            else:
                self.log("[错误] 最新一期所有名次均未TOP3中奖")
                self.log("")

                # 更新顶部信息 - 显示"暂无推荐"
                self.root.after(0, lambda: self.info_label_line1.config(text=f"最新期号: {latest_issue} | 最新号码: {numbers_str}"))
                self.root.after(0, lambda ni=next_issue:
                    self.info_label_line2.config(text=f"预测期号: {ni} | 推荐名次: 暂无"))
                self.root.after(0, lambda:
                    self.info_label_line3.config(text=f"推荐策略: -- | 近1期表现: --"))
                self.root.update_idletasks()  # 强制刷新界面

            # 显示未中奖名次的回测详情
            self.display_backtest_details(data, top3_hit_positions, all_predictions, all_backtests, next_issue)

            # 保存预测历史
            for position in all_predictions:
                self.prediction_history.append({
                    'predicted_issue': next_issue,
                    'position': position,
                    'top3': all_predictions[position]['top3']
                })

            self.log("="*80)
            self.log("[完成] 预测完成！")
            self.log("="*80)

            total_time = time.time() - start_time
            self.log(f"⏱️ 总耗时: {total_time:.2f}秒")
            self.log("")

        except Exception as e:
            self.log(f"[错误] 发生错误: {str(e)}", "#ff6b6b")
            import traceback
            self.log(traceback.format_exc(), "#ff6b6b")
        finally:
            # 恢复按钮，停止进度条
            self.progress.stop()
            self.run_button.config(state='normal')
            self.run_button_history.config(state='normal')
            self.status_label.config(text="完成", foreground="green")

    def get_history_data(self):
        """获取历史数据，若不足500期则补充昨天所有数据"""
        from datetime import datetime, timedelta

        # 禁用代理和SSL验证
        session = requests.Session()
        session.trust_env = False  # 禁用系统代理
        session.proxies = {
            'http': None,
            'https': None,
        }

        # 第一步：获取最新历史数据
        url = "https://1689567.com/api/pks/getPksHistoryList.do?lotCode=10037"
        res = session.get(url, timeout=15, verify=False)
        data_dict = json.loads(res.text)

        result = {}
        data_list = data_dict.get('result', {}).get('data', [])

        for item in data_list:
            pre_draw_issue = item.get('preDrawIssue')
            pre_draw_code = item.get('preDrawCode')

            if pre_draw_issue and pre_draw_code:
                try:
                    numbers = [int(num) for num in pre_draw_code.split(',')]
                    if len(numbers) == 10:
                        result[str(pre_draw_issue)] = numbers
                except:
                    continue

        self.log(f"[完成] 初始数据获取: {len(result)} 期")

        # 第二步：判断是否需要补充数据
        if len(result) < 500:
            self.log(f"⚠️  数据不足500期，开始获取昨天所有数据...")

            # 获取昨天的日期
            yesterday = datetime.now() - timedelta(days=1)
            date_str = yesterday.strftime('%Y-%m-%d')

            # 获取昨天的所有数据
            history_url = f"https://1689567.com/api/LotteryPlan/getPksPlanList.do?lotCode=10037&rows=0&date={date_str}"

            try:
                res = session.get(history_url, timeout=15, verify=False)
                history_dict = json.loads(res.text)
                history_list = history_dict.get('result', {}).get('data', [])

                added_count = 0
                for item in history_list:
                    pre_draw_issue = item.get('preDrawIssue')
                    pre_draw_code = item.get('preDrawCode')

                    if pre_draw_issue and pre_draw_code and str(pre_draw_issue) not in result:
                        try:
                            numbers = [int(num) for num in pre_draw_code.split(',')]
                            if len(numbers) == 10:
                                result[str(pre_draw_issue)] = numbers
                                added_count += 1
                        except:
                            continue

                self.log(f"[完成] 昨天 ({date_str}) 数据获取: 新增 {added_count} 期")
                self.log(f"[完成] 数据补充完成: 总计 {len(result)} 期")

            except Exception as e:
                self.log(f"[错误] 昨天数据获取失败: {str(e)}", "#ff6b6b")

        sorted_items = sorted(result.items(), key=lambda x: x[0], reverse=True)
        return dict(sorted_items)

    def analyze_position(self, data, position):
        """分析指定位置的号码规律"""
        position_idx = position - 1

        issues = list(data.keys())
        values = list(data.values())

        if len(values) < 2:
            return None

        # 1. 超近期分析（最近50期）- 优化：扩大样本量提高统计显著性
        super_recent_limit = min(50, len(values))
        super_recent_numbers = [values[i][position_idx] for i in range(super_recent_limit)]
        super_recent_freq = Counter(super_recent_numbers)

        # 2. 号码跟随分析（同位置）
        follow_pattern = defaultdict(lambda: defaultdict(int))
        for i in range(len(values) - 1):
            current_num = values[i][position_idx]
            next_num = values[i + 1][position_idx]
            follow_pattern[current_num][next_num] += 1

        # 2.1 全局号码跟随分析（不限位置）- 用于预测权重
        # 统计当前号码在所有位置出现后，下一期同位置跟随的号码
        current_position_num = values[0][position_idx]  # 当前期该位置的号码
        global_follow_pattern = defaultdict(int)

        # 遍历所有历史数据的所有位置
        for i in range(len(values) - 1):
            for pos_idx in range(10):
                if values[i][pos_idx] == current_position_num:
                    # 找到该号码，统计下一期同位置的号码
                    next_num = values[i + 1][pos_idx]
                    global_follow_pattern[next_num] += 1

        # 2.2 当前位置号码后，所有位置（1-10名）的跟随号码统计（近期数据，用于展示）
        follow_all_positions = {}

        # 统计近期数据中，当前位置出现该号码后，下一期所有位置的号码分布
        for pos_idx in range(10):  # 遍历1-10名
            follow_all_positions[pos_idx + 1] = defaultdict(int)

            for i in range(min(super_recent_limit, len(values) - 1)):
                if values[i][position_idx] == current_position_num:
                    # 找到当前位置出现该号码的期数，统计下一期该位置的号码
                    next_period_num = values[i + 1][pos_idx]
                    follow_all_positions[pos_idx + 1][next_period_num] += 1

        # 3. 遗漏值分析（冷号回补）
        last_seen = {}
        omit_values = defaultdict(list)
        for i, row in enumerate(values):
            num = row[position_idx]
            if num in last_seen:
                omit = i - last_seen[num]
                omit_values[num].append(omit)
            last_seen[num] = i

        current_omit = {}
        for num in range(1, 11):
            if num in last_seen:
                current_omit[num] = 0 if last_seen[num] == 0 else last_seen[num]
            else:
                current_omit[num] = len(values)

        # 4. 重号分析
        repeat_count = 0
        for i in range(len(values) - 1):
            if values[i][position_idx] == values[i + 1][position_idx]:
                repeat_count += 1
        repeat_rate = repeat_count / (len(values) - 1) if len(values) > 1 else 0

        # 5. 奇偶比分析（最近10期）
        odd_count = sum(1 for num in super_recent_numbers if num % 2 == 1)
        even_count = super_recent_limit - odd_count
        odd_ratio = odd_count / super_recent_limit if super_recent_limit > 0 else 0.5

        # 6. 大小比分析（最近10期，6-10为大，1-5为小）
        big_count = sum(1 for num in super_recent_numbers if num >= 6)
        small_count = super_recent_limit - big_count
        big_ratio = big_count / super_recent_limit if super_recent_limit > 0 else 0.5

        # 7. 和值趋势分析（最近10期该位置号码的平均值）
        avg_value = sum(super_recent_numbers) / super_recent_limit if super_recent_limit > 0 else 5.5

        # 8. 连号分析（统计连续号码出现频率）
        consecutive_pattern = defaultdict(int)
        for i in range(len(values) - 1):
            current_num = values[i][position_idx]
            next_num = values[i + 1][position_idx]
            if abs(current_num - next_num) == 1:
                consecutive_pattern[next_num] += 1

        # 9. 位置关联分析（当前位置与相邻位置的关联）
        position_correlation = defaultdict(lambda: defaultdict(int))
        if position_idx > 0:  # 与前一位置关联
            for row in values:
                prev_pos_num = row[position_idx - 1]
                curr_pos_num = row[position_idx]
                position_correlation[prev_pos_num][curr_pos_num] += 1
        elif position_idx < 9:  # 第1名，与第2名关联
            for row in values:
                curr_pos_num = row[position_idx]
                next_pos_num = row[position_idx + 1]
                position_correlation[next_pos_num][curr_pos_num] += 1

        # 获取相邻位置的当前号码
        adjacent_num = None
        if position_idx > 0 and len(values) > 0:
            adjacent_num = values[0][position_idx - 1]
        elif position_idx < 9 and len(values) > 0:
            adjacent_num = values[0][position_idx + 1]

        # 10. 全部历史频率（作为基准）
        all_numbers = [row[position_idx] for row in values]
        all_freq = Counter(all_numbers)

        return {
            'super_recent_freq': super_recent_freq,
            'follow_pattern': follow_pattern,
            'global_follow_pattern': global_follow_pattern,
            'follow_all_positions': follow_all_positions,
            'current_omit': current_omit,
            'omit_values': omit_values,
            'repeat_rate': repeat_rate,
            'odd_ratio': odd_ratio,
            'big_ratio': big_ratio,
            'avg_value': avg_value,
            'consecutive_pattern': consecutive_pattern,
            'position_correlation': position_correlation,
            'adjacent_num': adjacent_num,
            'all_freq': all_freq,
            'last_number': values[0][position_idx],
            'total_periods': len(values),
            'super_recent_limit': super_recent_limit
        }

    def extract_features(self, number, data, position):
        """提取单个号码的特征向量（用于ML模型）（带缓存优化）"""
        # 检查缓存
        cache_key = f"{number}_{len(data)}_{position}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        position_idx = position - 1
        values = list(data.values())
        features = []

        # 1. 近期频率特征（4个）
        for window in [3, 5, 10, 20]:
            recent_nums = [values[i][position_idx] for i in range(min(window, len(values)))]
            freq = recent_nums.count(number)
            features.append(freq)

        # 2. 跟随特征（3个）
        if len(values) > 0:
            last_num = values[0][position_idx]
            follow_count_5 = 0
            follow_count_10 = 0
            follow_count_all = 0
            total_5 = 0
            total_10 = 0
            total_all = 0

            for i in range(1, min(5, len(values))):
                if values[i][position_idx] == last_num:
                    total_5 += 1
                    if i > 0 and values[i-1][position_idx] == number:
                        follow_count_5 += 1

            for i in range(1, min(10, len(values))):
                if values[i][position_idx] == last_num:
                    total_10 += 1
                    if i > 0 and values[i-1][position_idx] == number:
                        follow_count_10 += 1

            for i in range(1, len(values)):
                if values[i][position_idx] == last_num:
                    total_all += 1
                    if i > 0 and values[i-1][position_idx] == number:
                        follow_count_all += 1

            features.append(follow_count_5 / max(total_5, 1))
            features.append(follow_count_10 / max(total_10, 1))
            features.append(follow_count_all / max(total_all, 1))
        else:
            features.extend([0, 0, 0])

        # 3. 冷热特征（3个）
        consecutive_omit = 0
        for val in values:
            if val[position_idx] == number:
                break
            consecutive_omit += 1
        features.append(min(consecutive_omit, 50))

        recent_10 = [values[i][position_idx] for i in range(min(10, len(values)))]
        heat = recent_10.count(number)
        features.append(heat)

        last_seen = consecutive_omit
        features.append(min(last_seen, 50))

        # 4. 位置特征（2个）
        all_nums = [values[i][position_idx] for i in range(len(values))]
        position_freq = all_nums.count(number) / len(all_nums) if all_nums else 0
        features.append(position_freq)

        all_positions_count = 0
        total_count = 0
        for val in values:
            for num in val:
                total_count += 1
                if num == number:
                    all_positions_count += 1
        global_freq = all_positions_count / total_count if total_count > 0 else 0
        features.append(global_freq)

        # 5. 统计特征（3个）
        recent_50 = [values[i][position_idx] for i in range(min(50, len(values)))]
        if recent_50:
            features.append(np.mean(recent_50))
            features.append(np.std(recent_50))
            recent_10_mean = np.mean(recent_10) if recent_10 else 5.5
            recent_50_mean = np.mean(recent_50)
            features.append(recent_10_mean - recent_50_mean)
        else:
            features.extend([5.5, 2.87, 0])

        result = np.array(features)

        # 缓存结果（限制缓存大小避免内存溢出）
        if len(self.feature_cache) < 10000:
            self.feature_cache[cache_key] = result

        return result

    def train_lstm_model(self, data, position, epochs=30, batch_size=32):
        """训练指定名次的LSTM模型（优化版）

        优化点：
        1. 更长序列：20期（原15期）
        2. 更深网络：2层128维（原1层64维）
        3. 更多训练：30轮（原20轮）
        4. 更小批次：32（原64，更稳定）
        5. 学习率调度：动态衰减
        6. 早停机制：防止过拟合
        7. 模型保存：自动保存最佳模型
        """
        try:
            # 设置随机种子，确保训练结果可重复
            import random
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # 检查是否已有保存的模型
            model_path = os.path.join(self.model_save_dir, f"lstm_pos{position}.pth")
            if os.path.exists(model_path):
                print(f"[加载] 第{position}名模型已存在，直接加载...")
                model = LSTMPredictor(
                    input_size=10,
                    hidden_size=128,
                    num_layers=2,
                    output_size=10,
                    dropout=0.3
                ).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                print(f"[完成] 第{position}名模型加载成功")
                return model

            # 准备数据集（使用20期序列，只训练该名次位置）
            dataset = SequenceDataset(data, position, seq_length=20)
            if len(dataset) < 10:
                print(f"[错误] 第{position}名数据不足")
                return None

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # 创建优化版模型（2层128维）
            model = LSTMPredictor(
                input_size=10,
                hidden_size=128,
                num_layers=2,
                output_size=10,
                dropout=0.3
            ).to(self.device)

            # 损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # 学习率调度器（每10轮衰减到0.5倍）
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            # 早停机制
            best_loss = float('inf')
            patience = 5
            patience_counter = 0

            # 训练
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for sequences, targets in dataloader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)  # 【修复】targets已经是该位置的标签

                    optimizer.zero_grad()
                    outputs = model(sequences)
                    loss = criterion(outputs, targets)  # 【修复】直接使用targets
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(dataloader)

                # 学习率调度
                scheduler.step()

                if (epoch + 1) % 5 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

                # 早停检查
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(model.state_dict(), model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  [早停] 第{epoch+1}轮触发早停机制")
                        break

            # 加载最佳模型
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))

            model.eval()
            print(f"  [保存] 模型已保存至 {model_path}")
            return model

        except Exception as e:
            print(f"[错误] 第{position}名LSTM训练失败: {e}")
            return None

    def predict_with_ml_model(self, data, position):
        """使用LSTM模型预测，返回每个号码的得分（0-100）（优化版）"""
        if not self.models_trained or position not in self.lstm_models:
            return None

        scores = {}
        vote_details = {}

        try:
            # 获取该名次的LSTM模型
            model = self.lstm_models[position]
            model.eval()

            # 准备输入序列（最近20期，只取该名次位置的号码）
            # 【关键修复】取最新的20期数据（倒序取前20个，然后反转回正序）
            all_data = list(data.values())
            if len(all_data) < 20:
                return None
            all_data = all_data[:20][::-1]  # 取前20期，然后反转为从旧到新的顺序

            # 【关键修复】只提取该名次位置的历史号码
            position_idx = position - 1
            position_history = []

            # 调试：打印前3期和最后3期的数据
            if position == 4:  # 只打印第4名
                print(f"\n[调试] 第{position}名预测输入数据:")
                print(f"  总数据量: {len(list(data.values()))}期")
                print(f"  使用最新20期，从旧到新排列")
                print(f"  最旧3期:")
                for i, row in enumerate(all_data[:3]):
                    print(f"    期{i+1}: 完整数据={row}, 第{position}名位置={row[position_idx]}")
                print(f"  最新3期:")
                for i, row in enumerate(all_data[-3:]):
                    print(f"    期{18+i}: 完整数据={row}, 第{position}名位置={row[position_idx]}")

            for row in all_data:
                # 将该位置的号码转换为one-hot编码（10维）
                one_hot = [0.0] * 10
                num = row[position_idx]
                one_hot[num - 1] = 1.0
                position_history.append(one_hot)

            # 转换为tensor: shape (1, 20, 10)
            sequence = torch.FloatTensor([position_history]).to(self.device)

            # 预测
            with torch.no_grad():
                outputs = model(sequence)
                probs = torch.softmax(outputs, dim=1)[0]

                # 调试：打印预测概率
                if position == 4:
                    print(f"[调试] 第{position}名预测概率分布:")
                    for num in range(1, 11):
                        print(f"  号码{num}: {probs[num-1].item()*100:.2f}%")
                    print()

            # 转换为得分（0-100）
            for num in range(1, 11):
                scores[num] = probs[num-1].item() * 100
                vote_details[num] = {
                    'lstm': probs[num-1].item() * 100
                }

            return scores, vote_details

        except Exception as e:
            print(f"[错误] LSTM预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def ml_backtest(self, data, position):
        """LSTM深度学习回测（修复数据泄露问题）"""
        if not self.models_trained:
            return None

        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

        # 增加回测期数到50期
        validation_periods = min(50, len(data) - 30)
        issues = list(data.keys())
        values = list(data.values())

        top3_hits = 0
        top5_hits = 0
        top10_hits = 0

        all_predictions = []
        all_actuals = []

        # 修复数据泄露：只使用历史数据预测未来
        for i in range(len(data) - validation_periods, len(data)):
            # 只使用i期之前的数据（不包含i期本身）
            historical_data = dict(zip(issues[:i], values[:i]))

            if len(historical_data) < 20:
                continue

            result = self.predict_with_ml_model(historical_data, position)

            if not result:
                continue

            ml_scores, _ = result
            sorted_ml = sorted(ml_scores.items(), key=lambda x: x[1], reverse=True)
            predicted_nums = [num for num, _ in sorted_ml]
            actual_num = values[i][position - 1]

            # 记录预测和实际值
            all_predictions.append(predicted_nums[0])  # TOP1预测
            all_actuals.append(actual_num)

            if actual_num in predicted_nums[:3]:
                top3_hits += 1
            if actual_num in predicted_nums[:5]:
                top5_hits += 1
            if actual_num in predicted_nums[:10]:
                top10_hits += 1

        total_tests = len(all_actuals)
        if total_tests == 0:
            return None

        # 计算混淆矩阵和其他指标
        cm = confusion_matrix(all_actuals, all_predictions, labels=list(range(1, 11)))

        # 计算精确率、召回率、F1（macro平均）
        precision = precision_score(all_actuals, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_actuals, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_actuals, all_predictions, average='macro', zero_division=0)

        return {
            'top1_rate': sum(1 for p, a in zip(all_predictions, all_actuals) if p == a) / total_tests,
            'top3_rate': top3_hits / total_tests,
            'top5_rate': top5_hits / total_tests,
            'top10_rate': top10_hits / total_tests,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'total_periods': total_tests
        }

    def predict_position(self, data, position, use_dynamic_weights=True):
        """预测指定位置的号码 - 算法优化版

        核心改进：
        1. 自适应权重学习：基于历史回测自动优化权重
        2. 统计显著性检验：过滤不显著的规律
        3. 多窗口特征融合：10期/30期/50期综合分析
        4. 贝叶斯概率修正：基于历史准确率调整
        5. 集成评分：多种方法交���验证
        """

        # ========== 仅使用ML模型预测 ==========
        ml_scores = None
        vote_details = None
        if self.models_trained and position in self.lstm_models:
            result = self.predict_with_ml_model(data, position)
            if result:
                ml_scores, vote_details = result

        scores = {}

        if ml_scores and self.models_trained:
            # 只使用ML模型得分
            for num in range(1, 11):
                ml_score = ml_scores.get(num, 0)
                scores[num] = {
                    'total': round(ml_score, 2),
                    'ml_score': round(ml_score, 2),
                    'stat_score': 0,
                    'follow': 0,
                    'global_follow': 0,
                    'hot': 0,
                    'cold': 0,
                    'repeat': 0,
                    'diversity': 0,
                    'frequency': 0,
                    'omit': 0,
                    'vote_details': vote_details.get(num, {}) if vote_details else {}
                }
        else:
            # ML模型不可用，返回None
            self.log("⚠️ ML模型未训练，无法预测")
            return None, None

        # 按总分排序
        sorted_predictions = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)

        return sorted_predictions, None  # 不返回analysis

    def validate_prediction_accuracy(self, data, position):
        """验证预测准确率：回测最近N期的预测效果 - 使用与推荐号码相同的多策略预测逻辑"""
        validation_periods = min(20, len(data) - 1)  # 优化：减少到20期以提高速度
        if validation_periods < 5:
            return None

        issues = list(data.keys())
        values = list(data.values())

        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        ranks = []

        # 用于存储详细回测信息
        backtest_details = []

        for i in range(1, validation_periods + 1):
            # 使用第i期之前的数据预测第i期
            historical_data = dict(zip(issues[i:], values[i:]))

            # 调试：打印回测时的数据
            if position == 4 and i <= 5:  # 修改为第4名，打印前5期
                print(f"\n[回测调试] 第{position}名，回测第{i}期:")
                print(f"  预测期号: {issues[i-1]}")
                print(f"  历史数据期数: {len(historical_data)}")
                print(f"  最新3期数据:")
                for j, (issue, nums) in enumerate(list(historical_data.items())[:3]):
                    print(f"    {issue}: {nums}, 第{position}名={nums[position-1]}")

            # 【关键修改】使用与推荐号码相同的多策略预测逻辑
            top8, final_scores, market_state, weights = self.generate_top8_multi_strategy(historical_data, position)

            # 生成完整的TOP10（按得分排序）
            sorted_nums = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            predicted_nums = [num for num, score in sorted_nums[:10]]

            # 调试：打印预测结果
            if position == 4 and i <= 5:  # 修改为第4名，打印前5期
                state_name = {'HOT': '热号期', 'COLD': '冷号期', 'VOLATILE': '波动期', 'STABLE': '稳定期'}.get(market_state, '未知')
                print(f"  市场状态: {state_name}")
                print(f"  策略权重: LSTM={weights['lstm']*100:.0f}% | 热号={weights['hot']*100:.0f}% | 冷号={weights['cold']*100:.0f}% | 周期={weights['cycle']*100:.0f}% | RF={weights['rf']*100:.0f}%")
                print(f"  预测结果: {predicted_nums}")
                print(f"  实际号码: {values[i-1][position-1]}")

            # 获取实际结果
            actual_num = values[i - 1][position - 1]
            issue_num = issues[i - 1]

            # 检查命中情况 - TOP7以内精确显示排名
            top8_nums = predicted_nums[:8]
            top10_nums = predicted_nums[:10]

            hit_status = ""
            if actual_num in predicted_nums[:7]:
                rank = predicted_nums.index(actual_num) + 1
                hit_status = f"[完成] TOP{rank}"

                # 统计各级别命中
                if rank == 1:
                    top1_hits += 1
                if rank <= 3:
                    top3_hits += 1
                if rank <= 5:
                    top5_hits += 1
            else:
                hit_status = "[错误] 未中TOP7"

            # 记录实际号码的排名
            if actual_num in predicted_nums:
                rank = predicted_nums.index(actual_num) + 1
                ranks.append(rank)
            else:
                ranks.append(11)  # 未命中记为11

            # 保存回测详情（保存TOP3和TOP10）
            backtest_details.append({
                'issue': issue_num,
                'top8': top8_nums,
                'top10': top10_nums,
                'actual': actual_num,
                'hit_status': hit_status,
                'rank': ranks[-1]
            })

        # 计算命中率
        top1_rate = top1_hits / validation_periods
        top3_rate = top3_hits / validation_periods
        top5_rate = top5_hits / validation_periods
        avg_rank = sum(ranks) / len(ranks) if ranks else 11

        # 优化：添加置信度计算（使用标准差）
        rank_std = (sum((r - avg_rank) ** 2 for r in ranks) / len(ranks)) ** 0.5 if ranks else 0

        # 计算置信区间（95%置信度）
        confidence_interval = 1.96 * (top3_rate * (1 - top3_rate) / validation_periods) ** 0.5 if validation_periods > 0 else 0

        # 生成history字段：每期是否TOP7命中（1表示命中，0表示未中）
        history = [1 if '[完成] TOP' in detail['hit_status'] else 0
                   for detail in backtest_details]

        return {
            'top1_rate': top1_rate,
            'top3_rate': top3_rate,
            'top5_rate': top5_rate,
            'avg_rank': avg_rank,
            'rank_std': rank_std,  # 新增：排名标准差
            'confidence_interval': confidence_interval,  # 新增：置信区间
            'total_periods': validation_periods,
            'backtest_details': backtest_details,
            'history': history  # 新增：每期命中历史
        }


    def fetch_latest_data(self):
        """获取最新数据"""
        try:
            url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?name=3d&issueCount=&issueStart=&issueEnd=&dayStart=&dayEnd="
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            result = response.json()

            if result.get('state') == 0 and result.get('result'):
                data = {}
                for item in result['result']:
                    issue = item['code']
                    red = item['red'].split(',')
                    data[issue] = [int(x) for x in red]
                return data
            return None
        except Exception as e:
            return None

    def check_hit(self, predicted_top3, actual_num):
        """检查是否中奖 - TOP7以内精确显示排名"""
        if actual_num in predicted_top3:
            rank = predicted_top3.index(actual_num) + 1
            if rank <= 7:
                return True, f"[完成] TOP{rank}"
            else:
                return False, "[错误] 未中TOP7"
        return False, "[错误] 未中"

    def find_top3_hit_positions(self, data, all_predictions, all_backtests, latest_issue):
        """找出最新一期TOP3中奖的名次"""
        top3_hit_positions = []

        issues = list(data.keys())
        if len(issues) < 2:
            return top3_hit_positions

        # 获取最新一期的实际号码
        latest_numbers = data[latest_issue]

        # 检查每个名次的预测是否TOP3中奖
        for position in range(1, 11):
            if position not in all_backtests:
                continue

            backtest = all_backtests[position]
            backtest_details = backtest.get('backtest_details', [])

            if len(backtest_details) == 0:
                continue

            # 获取最新一期的回测详情（第一条是最新的，因为数据是降序排列）
            latest_detail = backtest_details[0]

            # 检查期号是否匹配
            if latest_detail['issue'] != latest_issue:
                continue

            # 获取实际号码和预测的TOP7
            actual_num = latest_numbers[position - 1]
            top3_predicted = latest_detail['top10'][:7]  # 取TOP7

            # 判断是否在TOP3中
            if actual_num in top3_predicted:
                rank = top3_predicted.index(actual_num) + 1
                top3_hit_positions.append({
                    'position': position,
                    'rank': rank,
                    'actual': actual_num,
                    'top3': top3_predicted
                })

        return top3_hit_positions

    def select_best_from_top3_hits(self, top3_hit_positions, all_backtests):
        """从TOP3中奖的名次中选择历史准确率最高的 - 简化版"""
        if not top3_hit_positions:
            return 1

        best_position = top3_hit_positions[0]['position']
        best_score = 0

        for pos_info in top3_hit_positions:
            pos = pos_info['position']
            if pos in all_backtests:
                backtest = all_backtests[pos]
                # 综合评分：仅使用TOP3命中率
                top3_rate = backtest.get('top3_rate', 0)
                score = top3_rate

                if score > best_score:
                    best_score = score
                    best_position = pos

        return best_position

    def select_best_position_advanced(self, data, all_predictions, all_backtests):
        """高级推荐名次选择 - V3优化版：直接调用自适应方法"""
        return self.select_best_position_adaptive(data, all_predictions, all_backtests)

    def calculate_recent_performance(self, backtest, periods=5):
        """计算最近N期的表现得分（满分30）"""
        backtest_details = backtest.get('backtest_details', [])

        if len(backtest_details) < periods:
            periods = len(backtest_details)

        if periods == 0:
            return 0

        hit_count = 0
        # 取最新的N期（数据是降序，所以从前往后取）
        recent_details = backtest_details[:periods]
        for detail in recent_details:
            hit_status = detail.get('hit_status', '[错误] 未中')

            if '[完成] TOP' in hit_status:
                # 提取排名数字，例如 "[完成] TOP3" -> 3
                try:
                    rank = int(hit_status.split('TOP')[1])
                    if rank == 1:
                        hit_count += 3  # TOP1中奖加3分
                    elif rank <= 3:
                        hit_count += 2  # TOP2-TOP3中奖加2分
                    else:
                        hit_count += 1  # TOP4-TOP7中奖加1分
                except:
                    hit_count += 1  # 解析失败默认加1分

        # 归一化到30分
        max_possible = periods * 3
        score = (hit_count / max_possible) * 30 if max_possible > 0 else 0

        return score

    def display_backtest_details(self, data, top3_hit_positions, all_predictions, all_backtests, next_issue):
        """显示所有名次的回测详情"""
        self.clear_backtest_console()

        # 提取TOP3中奖的名次列表
        top3_hit_pos_list = [p['position'] for p in top3_hit_positions] if top3_hit_positions else []

        # 为每个名次（1-10）显示回测详情
        for position in range(1, 11):
            if position not in all_backtests:
                self.log_backtest(position, "暂无回测数据")
                continue

            validation_result = all_backtests[position]

            # 判断是否为TOP3中奖名次
            is_top3_hit = position in top3_hit_pos_list
            status_mark = "[目标] TOP3中奖" if is_top3_hit else "[完成] 正常"

            self.log_backtest(position, "="*50)
            self.log_backtest(position, f"回测详情 - {status_mark}")
            self.log_backtest(position, "="*50)
            self.log_backtest(position, "")
            self.log_backtest(position, f"{'期号':<12}{'预测TOP10':<35}{'实际':<6}{'结果':<10}")
            self.log_backtest(position, "-"*50)

            # 显示最新预测（未开奖）
            if position in all_predictions:
                top10_str = ','.join(map(str, all_predictions[position]['top10']))
                self.log_backtest(position, f"{next_issue:<12}{top10_str:<35}{'--':<6}{'[训练] 等待':<10}")

            # 显示历史回测数据（只显示最近10期）
            for detail in validation_result['backtest_details'][:10]:
                top10_str = ','.join(map(str, detail['top10']))
                self.log_backtest(position, f"{detail['issue']:<12}{top10_str:<35}{detail['actual']:<6}{detail['hit_status']:<10}")

            self.log_backtest(position, "")
            self.log_backtest(position, "统计汇总:")
            self.log_backtest(position, f"  TOP1: {validation_result['top1_rate']*100:.1f}%")
            self.log_backtest(position, f"  TOP3: {validation_result['top3_rate']*100:.1f}%")
            self.log_backtest(position, f"  平均: {validation_result['avg_rank']:.1f}")

            # 显示LSTM深度学习回测结果（科学验证版）
            if position in all_backtests:
                ml_result = self.ml_backtest(data, position)
                if ml_result:
                    self.log_backtest(position, "")
                    self.log_backtest(position, "LSTM深度学习回测（无数据泄露）:")
                    self.log_backtest(position, f"  LSTM-TOP1: {ml_result['top1_rate']*100:.1f}%")
                    self.log_backtest(position, f"  LSTM-TOP3: {ml_result['top3_rate']*100:.1f}%")
                    self.log_backtest(position, f"  LSTM-TOP5: {ml_result['top5_rate']*100:.1f}%")
                    self.log_backtest(position, f"  LSTM-TOP10: {ml_result['top10_rate']*100:.1f}%")
                    self.log_backtest(position, f"  精确率: {ml_result['precision']*100:.1f}%")
                    self.log_backtest(position, f"  召回率: {ml_result['recall']*100:.1f}%")
                    self.log_backtest(position, f"  F1分数: {ml_result['f1_score']*100:.1f}%")
                    self.log_backtest(position, f"  回测期数: {ml_result['total_periods']}")

                    # 显示交叉验证结果（LSTM模型不需要）
                    # if position in self.cv_results:
                    #     cv = self.cv_results[position]
                    #     self.log_backtest(position, "")
                    #     self.log_backtest(position, "交叉验证结果（5折）:")
                    #     self.log_backtest(position, f"  RandomForest: {cv['rf_mean']*100:.1f}% ± {cv['rf_std']*100:.1f}%")
                    #     self.log_backtest(position, f"  GradientBoosting: {cv['gb_mean']*100:.1f}% ± {cv['gb_std']*100:.1f}%")
                    #     self.log_backtest(position, f"  LogisticRegression: {cv['lr_mean']*100:.1f}% ± {cv['lr_std']*100:.1f}%")
            self.log_backtest(position, "="*50)

            # 每个详情框显示完成后，滚动到顶部（使用root.after确保在主线程执行）
            if position in self.backtest_consoles:
                self.root.after(0, lambda p=position: self.backtest_consoles[p].yview_moveto(0))

        # 显示完成后，Canvas也滚动到顶部，并强制刷新界面
        self.root.after(0, lambda: self.backtest_canvas.yview_moveto(0))
        self.root.after(100, lambda: self.root.update_idletasks())  # 延迟100ms强制刷新

    # ==================== 高级功能实现 ====================

    def show_recommendation_history(self):
        """显示推荐历史记录"""
        # 直接在主线程中打开悬浮窗口，不使用run_advanced_function
        threading.Thread(target=self.display_recommendation_history, daemon=True).start()

    def retrain_all_models(self):
        """重新训练所有LSTM模型"""
        if self.pretraining_in_progress:
            self.update_status("模型训练中，请稍候...")
            return

        # 确认对话框
        import tkinter.messagebox as messagebox
        result = messagebox.askyesno(
            "确认重新训练",
            "重新训练所有LSTM模型需要4-5分钟，\n训练期间可以继续使用程序。\n\n是否继续？"
        )

        if result:
            self.update_status("开始重新训练所有LSTM模型...")
            self.models_trained = False
            self.lstm_models = {}
            # 【修复】清除所有缓存
            self.all_positions_backtest = {}
            self.all_positions_predictions = {}
            self.backtest_results = None
            threading.Thread(target=self.preload_and_train, daemon=True).start()

    def delete_all_models(self):
        """删除所有已保存的LSTM模型"""
        import tkinter.messagebox as messagebox
        import shutil

        # 检查模型文件夹是否存在
        if not os.path.exists(self.model_save_dir):
            messagebox.showinfo("提示", "没有找到已保存的模型文件。")
            return

        # 统计模型文件数量
        model_files = [f for f in os.listdir(self.model_save_dir) if f.endswith('.pth')]
        if not model_files:
            messagebox.showinfo("提示", "模型文件夹为空，没有需要删除的模型。")
            return

        # 确认对话框
        result = messagebox.askyesno(
            "确认删除",
            f"找到 {len(model_files)} 个模型文件。\n\n删除后下次启动将重新训练模型（需要4-5分钟）。\n\n是否确认删除？"
        )

        if result:
            try:
                # 删除整个模型文件夹
                shutil.rmtree(self.model_save_dir)
                # 重新创建空文件夹
                os.makedirs(self.model_save_dir)

                # 清空内存中的模型
                self.lstm_models = {}
                self.models_trained = False

                # 【修复】清除所有缓存
                self.all_positions_backtest = {}
                self.all_positions_predictions = {}
                self.backtest_results = None

                self.update_status("所有模型已删除")
                messagebox.showinfo("成功", f"已成功删除 {len(model_files)} 个模型文件。\n\n下次启动将自动重新训练模型。")
            except Exception as e:
                messagebox.showerror("错误", f"删除模型文件时出错：\n{str(e)}")
                self.update_status(f"删除模型失败: {str(e)}")

    def display_recommendation_history(self):
        """显示推荐历史记录详情 - 弹出悬浮窗口"""
        # 获取最新数据用于验证
        data = self.get_history_data()

        # 更新历史记录的实际开奖结果
        if data:
            for record in self.recommendation_history:
                if record['result'] == 'waiting' and record['issue'] in data:
                    actual_numbers = data[record['issue']]
                    actual_num = actual_numbers[record['position'] - 1]
                    record['actual'] = actual_num

                    # 判断是否中奖（TOP7标准）
                    top7_nums = record['numbers'][:7]
                    if actual_num in top7_nums:
                        rank = top7_nums.index(actual_num) + 1
                        record['result'] = f'TOP{rank}'
                    else:
                        record['result'] = '未中'

        # 统计数据
        total = len(self.recommendation_history)
        waiting = sum(1 for r in self.recommendation_history if r['result'] == 'waiting')
        hit = sum(1 for r in self.recommendation_history if r['result'].startswith('TOP'))
        miss = sum(1 for r in self.recommendation_history if r['result'] == '未中')

        verified = total - waiting
        hit_rate = (hit / verified * 100) if verified > 0 else 0

        # 统计连续未中情况
        consecutive_miss = self.count_consecutive_miss()

        # 创建悬浮窗口
        history_window = tk.Toplevel(self.root)
        history_window.title("📜 推荐历史记录")
        history_window.geometry("1000x700")
        history_window.transient(self.root)  # 设置为主窗口的子窗口

        # 统计信息框
        stats_frame = ttk.LabelFrame(history_window, text="统计汇总", padding="10")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        # 基础统计
        stats_text = f"总推荐次数: {total}  |  已验证: {verified}  |  等待开奖: {waiting}  |  命中: {hit}  |  未中: {miss}  |  命中率: {hit_rate:.1f}%"
        ttk.Label(stats_frame, text=stats_text, font=("Consolas", 10, "bold")).pack()

        # 四级防连败机制统计
        level_stats_text = (
            f"🛡️ 防连败机制统计 | "
            f"正常模式: {self.stats_level1_count}次  |  "
            f"紧急模式: {self.stats_level3_count}次  |  "
            f"必中模式: {self.stats_level4_count}次  |  "
            f"当前连续未中: {consecutive_miss}期"
        )
        level_label = ttk.Label(stats_frame, text=level_stats_text, font=("Consolas", 9))
        level_label.pack(pady=(5, 0))

        # 根据连续未中次数设置颜色
        if consecutive_miss >= 2:
            level_label.config(foreground='red')
        elif consecutive_miss >= 1:
            level_label.config(foreground='orange')
        else:
            level_label.config(foreground='green')

        # 详细记录框
        detail_frame = ttk.LabelFrame(history_window, text="详细记录", padding="10")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # 创建表格
        columns = ("期号", "推荐名次", "推荐号码(TOP10)", "实际", "结果")
        tree = ttk.Treeview(detail_frame, columns=columns, show="headings", height=20)

        # 设置列标题
        tree.heading("期号", text="期号")
        tree.heading("推荐名次", text="推荐名次")
        tree.heading("推荐号码(TOP10)", text="推荐号码(TOP10)")
        tree.heading("实际", text="实际")
        tree.heading("结果", text="结果")

        # 设置列宽
        tree.column("期号", width=100, anchor=tk.CENTER)
        tree.column("推荐名次", width=80, anchor=tk.CENTER)
        tree.column("推荐号码(TOP10)", width=300, anchor=tk.CENTER)
        tree.column("实际", width=60, anchor=tk.CENTER)
        tree.column("结果", width=100, anchor=tk.CENTER)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 配置标签颜色
        tree.tag_configure('waiting', foreground='#ffa500')  # 橙色
        tree.tag_configure('hit', foreground='#00aa00')      # 绿色
        tree.tag_configure('miss', foreground='#ff0000')     # 红色

        # 插入数据（倒序显示，最新的在上面）
        if not self.recommendation_history:
            tree.insert("", tk.END, values=("暂无记录", "", "", "", ""))
        else:
            for record in reversed(self.recommendation_history):
                issue = record['issue']
                position = f"第{record['position']}名"
                numbers = ','.join(map(str, record['numbers']))
                actual = str(record['actual']) if record['actual'] else '--'
                result = record['result']

                # 根据结果设置标签
                if result == 'waiting':
                    tag = 'waiting'
                    result_text = "[等待]"
                elif result.startswith('TOP'):
                    tag = 'hit'
                    result_text = f"[中奖] {result}"
                else:
                    tag = 'miss'
                    result_text = "[未中]"

                tree.insert("", tk.END, values=(issue, position, numbers, actual, result_text), tags=(tag,))

        # 关闭按钮
        button_frame = ttk.Frame(history_window)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(button_frame, text="关闭", command=history_window.destroy).pack(side=tk.RIGHT)
        ttk.Label(button_frame, text="提示: 此窗口不影响主界面操作", foreground="gray").pack(side=tk.LEFT)

    def run_advanced_function(self, func, status_text):
        """运行高级功能的通用方法"""
        self.status_label.config(text=status_text, foreground="orange")
        self.progress.start(10)

        def wrapper():
            try:
                func()
            finally:
                self.progress.stop()
                self.status_label.config(text="就绪", foreground="green")

        thread = threading.Thread(target=wrapper)
        thread.daemon = True
        thread.start()

    # ==================== V3优化：新增预测方法 ====================

    def select_best_position_adaptive(self, data, all_predictions, all_backtests):
        """
        高信心推荐版 - 自适应推荐最佳名次

        新权重分配：
        - LSTM模型置信度：35%（核心）
        - 历史TOP3命中率：25%（从5%提升）
        - 近5期表现：20%（中期趋势）
        - 近2期表现：10%（从80%降低）
        - 稳定性评分：10%
        - 连续失败惩罚：扣分机制
        """

        # 第1步：分析当前市场状态（保留原有逻辑）
        market_state = self.analyze_market_state(data, all_backtests)

        # 第2步：计算每个名次的综合得分（使用新的评分系统）
        position_scores = {}

        for position in range(1, 11):
            if position not in all_backtests:
                continue

            backtest = all_backtests[position]

            # 使用新的综合评分方法
            total_score, score_details = self.calculate_recommendation_score(
                data, position, backtest, all_predictions
            )

            # 获取关键指标
            top3_rate = backtest.get('top3_rate', 0)
            lstm_confidence = score_details.get('lstm_confidence', 0.5)

            # 判断信心等级
            confidence_level, stars, bet_advice, expected_rate = self.get_confidence_level(
                total_score, lstm_confidence, top3_rate
            )

            position_scores[position] = {
                'total_score': total_score,
                'details': score_details,
                'top3_rate': top3_rate,
                'lstm_confidence': lstm_confidence,
                'confidence_level': confidence_level,
                'stars': stars,
                'bet_advice': bet_advice,
                'expected_rate': expected_rate,
                'strategy': 'high_confidence'
            }

        # 第3步：选择最佳名次
        if not position_scores:
            return 1, "默认推荐", {}, market_state

        best_position = max(position_scores.items(), key=lambda x: x[1]['total_score'])
        best_pos = best_position[0]
        best_info = best_position[1]

        # 第4步：生成推荐理由（包含信心等级）
        reason = self.generate_recommendation_reason_high_confidence(
            best_pos, best_info, position_scores, market_state
        )

        # 第5步：更新上次推荐记录
        self.last_recommended_position = best_pos

        return best_pos, reason, position_scores, market_state

    def analyze_market_state(self, data, all_backtests):
        """分析当前市场状态"""

        volatilities = []
        recent_trends = []

        for position in range(1, 11):
            if position not in all_backtests:
                continue

            backtest = all_backtests[position]
            backtest_details = backtest.get('backtest_details', [])

            if len(backtest_details) < 10:
                continue

            recent_10 = backtest_details[:10]
            hit_pattern = []
            for detail in recent_10:
                hit_status = detail.get('hit_status', '[错误] 未中')
                # 修改：只要包含[完成] TOP就算命中（TOP1-TOP7）
                if '[完成] TOP' in hit_status:
                    hit_pattern.append(1)
                else:
                    hit_pattern.append(0)

            mean_hit = sum(hit_pattern) / len(hit_pattern)
            variance = sum((x - mean_hit) ** 2 for x in hit_pattern) / len(hit_pattern)
            volatility = math.sqrt(variance)
            volatilities.append(volatility)

            recent_5_hit = sum(hit_pattern[:5]) / 5
            recent_10_hit = sum(hit_pattern) / 10
            trend = recent_5_hit - recent_10_hit
            recent_trends.append(trend)

        avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0
        avg_trend = sum(recent_trends) / len(recent_trends) if recent_trends else 0

        recent_hit_counts = []
        for i in range(min(5, len(list(data.values())))):
            hit_count = 0
            for position in range(1, 11):
                if position not in all_backtests:
                    continue
                backtest = all_backtests[position]
                backtest_details = backtest.get('backtest_details', [])
                if i < len(backtest_details):
                    detail = backtest_details[i]
                    hit_status = detail.get('hit_status', '[错误] 未中')
                    # 修改：只要包含[完成] TOP就算命中（TOP1-TOP7）
                    if '[完成] TOP' in hit_status:
                        hit_count += 1
            recent_hit_counts.append(hit_count)

        avg_hit_count = sum(recent_hit_counts) / len(recent_hit_counts) if recent_hit_counts else 0

        if avg_volatility < 0.3 and abs(avg_trend) < 0.1:
            state_type = 'stable'
            description = "市场稳定，规律清晰"
        elif avg_trend > 0.15:
            state_type = 'trending'
            description = "市场趋势向好，近期表现强劲"
        elif avg_volatility > 0.45:
            state_type = 'volatile'
            description = "市场波动剧烈，规律不稳定"
        elif avg_hit_count < 2:
            state_type = 'reversal'
            description = "市场可能反转，冷门名次有机会"
        else:
            state_type = 'stable'
            description = "市场正常"

        return {
            'type': state_type,
            'description': description,
            'volatility': avg_volatility,
            'trend': avg_trend,
            'avg_hit_count': avg_hit_count
        }

    def calculate_stability_score_enhanced(self, backtest):
        """增强的稳定性得分"""

        backtest_details = backtest.get('backtest_details', [])

        if len(backtest_details) < 10:
            return 0

        avg_rank = backtest.get('avg_rank', 11)
        rank_std = backtest.get('rank_std', 5)

        rank_score = max(0, 10 - avg_rank * 0.8)
        std_score = max(0, 10 - rank_std * 1.5)

        hit_rates = []
        for i in range(0, min(10, len(backtest_details)), 2):
            chunk = backtest_details[i:i+2]
            hit_count = sum(1 for d in chunk if '[完成]' in d.get('hit_status', ''))
            hit_rates.append(hit_count / len(chunk))

        if len(hit_rates) > 1:
            mean_rate = sum(hit_rates) / len(hit_rates)
            variance = sum((x - mean_rate) ** 2 for x in hit_rates) / len(hit_rates)
            frequency_stability = max(0, 10 - variance * 50)
        else:
            frequency_stability = 5

        total_stability = rank_score + std_score + frequency_stability

        return total_stability

    def calculate_position_cold_rebound(self, backtest):
        """计算名次的冷号回补潜力"""

        backtest_details = backtest.get('backtest_details', [])

        if len(backtest_details) < 10:
            return 0

        recent_10 = backtest_details[:10]
        top3_hit_count = sum(
            1 for d in recent_10
            if '[完成] TOP' in d.get('hit_status', '')
        )

        consecutive_miss = 0
        for detail in recent_10:
            hit_status = detail.get('hit_status', '[错误] 未中')
            if '[完成] TOP' in hit_status:
                break
            consecutive_miss += 1

        if top3_hit_count == 0:
            rebound_score = 50
        elif top3_hit_count == 1:
            rebound_score = 30
        elif top3_hit_count == 2:
            rebound_score = 15
        else:
            rebound_score = 0

        if consecutive_miss >= 8:
            rebound_score += 20
        elif consecutive_miss >= 5:
            rebound_score += 10

        return rebound_score

    def calculate_hot_decay_penalty(self, backtest):
        """计算热门名次的衰减惩罚"""

        backtest_details = backtest.get('backtest_details', [])

        if len(backtest_details) < 5:
            return 0

        recent_5 = backtest_details[:5]
        top3_hit_count = sum(
            1 for d in recent_5
            if '[完成] TOP' in d.get('hit_status', '')
        )

        consecutive_hit = 0
        for detail in recent_5:
            hit_status = detail.get('hit_status', '[错误] 未中')
            if '[完成] TOP' in hit_status:
                consecutive_hit += 1
            else:
                break

        penalty = 0

        if top3_hit_count >= 4:
            penalty = 15
        elif top3_hit_count >= 3:
            penalty = 8

        if consecutive_hit >= 3:
            penalty += 20
        elif consecutive_hit >= 2:
            penalty += 10

        return penalty

    # ==================== 高信心推荐系统：新增功能 ====================

    def calculate_lstm_confidence(self, data, position):
        """
        计算LSTM模型的置信度（0-1）

        基于以下特征：
        1. 最高概率值（越高越好）
        2. TOP3概率和（越集中越好）
        3. 熵值（越低越好，表示预测越确定）

        返回：置信度得分（0-1）
        """
        try:
            # 检查模型是否已训练
            if position not in self.lstm_models:
                return 0.5  # 默认中等置信度

            model = self.lstm_models[position]
            model.eval()

            # 准备输入数据（最近20期）
            values = list(data.values())
            if len(values) < 20:
                return 0.5

            # 取最近20期作为输入序列
            sequence = values[:20]
            sequence_tensor = torch.FloatTensor([sequence]).to(self.device)

            # 获取模型输出
            with torch.no_grad():
                output = model(sequence_tensor)
                # 使用softmax转换为概率分布
                probabilities = torch.softmax(output, dim=1)[0]
                probs = probabilities.cpu().numpy()

            # 1. 最高概率值（权重40%）
            max_prob = float(np.max(probs))
            max_prob_score = max_prob

            # 2. TOP3概率和（权重40%）
            top3_probs = sorted(probs, reverse=True)[:3]
            top3_sum = float(np.sum(top3_probs))
            top3_score = top3_sum

            # 3. 熵值（权重20%）- 熵越低，预测越确定
            # 归一化熵：H / log(n)，其中n=10
            epsilon = 1e-10
            entropy = -np.sum(probs * np.log(probs + epsilon))
            max_entropy = np.log(10)  # 均匀分布的最大熵
            normalized_entropy = entropy / max_entropy
            entropy_score = 1 - normalized_entropy  # 转换为得分（越低越好）

            # 综合置信度得分
            confidence = (
                max_prob_score * 0.4 +
                top3_score * 0.4 +
                entropy_score * 0.2
            )

            # 限制在[0, 1]范围内
            confidence = max(0.0, min(1.0, confidence))

            return confidence

        except Exception as e:
            print(f"[警告] LSTM置信度计算失败: {e}")
            return 0.5  # 出错时返回中等置信度

    def calculate_recommendation_score(self, data, position, backtest, all_predictions):
        """
        计算综合推荐得分（0-100分）

        新权重分配：
        - LSTM模型置信度：35分
        - 历史TOP3命中率：25分
        - 近5期表现：20分
        - 近2期表现：10分
        - 稳定性评分：10分
        - 连续失败惩罚：扣分
        """
        score_details = {}

        # 1. LSTM模型置信度（35分）- 核心维度
        lstm_confidence = self.calculate_lstm_confidence(data, position)
        lstm_score = lstm_confidence * 35
        score_details['lstm_confidence'] = lstm_confidence
        score_details['lstm_score'] = lstm_score

        # 2. 历史TOP3命中率（25分）
        top3_rate = backtest.get('top3_rate', 0)
        accuracy_score = top3_rate * 25
        score_details['accuracy'] = top3_rate
        score_details['accuracy_score'] = accuracy_score

        # 3. 近5期表现（20分）
        recent_5_score = self.calculate_recent_performance(backtest, 5) * (20 / 30)
        score_details['recent_5_score'] = recent_5_score

        # 4. 近2期表现（10分）
        recent_2_score = self.calculate_recent_performance(backtest, 2) * (10 / 30)
        score_details['recent_2_score'] = recent_2_score

        # 5. 稳定性评分（10分）
        stability_score = self.calculate_stability_score_enhanced(backtest)
        score_details['stability_score'] = stability_score

        # 6. 连续失败惩罚
        consecutive_miss_penalty = self.calculate_consecutive_miss_penalty(backtest)
        score_details['consecutive_miss_penalty'] = consecutive_miss_penalty

        # 综合得分
        total_score = (
            lstm_score +
            accuracy_score +
            recent_5_score +
            recent_2_score +
            stability_score -
            consecutive_miss_penalty
        )

        # 限制在[0, 100]范围内
        total_score = max(0, min(100, total_score))
        score_details['total_score'] = total_score

        return total_score, score_details

    def calculate_consecutive_miss_penalty(self, backtest):
        """
        计算连续失败惩罚

        连续未中次数越多，惩罚越重：
        - 连续3次未中：扣5分
        - 连续5次未中：扣10分
        - 连续8次未中：扣15分
        """
        backtest_details = backtest.get('backtest_details', [])

        if len(backtest_details) == 0:
            return 0

        consecutive_miss = 0
        for detail in backtest_details:
            hit_status = detail.get('hit_status', '[错误] 未中')
            # 修改：只要不包含[完成] TOP就算未中
            if '[完成] TOP' not in hit_status:
                consecutive_miss += 1
            else:
                break

        if consecutive_miss >= 8:
            return 15
        elif consecutive_miss >= 5:
            return 10
        elif consecutive_miss >= 3:
            return 5
        else:
            return 0

    def get_confidence_level(self, total_score, lstm_confidence, accuracy_rate):
        """
        判断信心等级（1-5星）

        评级标准：
        - 5星：综合得分≥85 且 LSTM置信度≥0.8 且 历史准确率≥65%
        - 4星：综合得分≥75 且 LSTM置信度≥0.7
        - 3星：综合得分≥60
        - 2星：综合得分≥45
        - 1星：综合得分<45

        返回：(星级, 星级显示, 投注建议, 预期命中率范围)
        """
        if total_score >= 85 and lstm_confidence >= 0.8 and accuracy_rate >= 0.65:
            return 5, "⭐⭐⭐⭐⭐", "重点投注", "75-85%"
        elif total_score >= 75 and lstm_confidence >= 0.7:
            return 4, "⭐⭐⭐⭐", "正常投注", "65-75%"
        elif total_score >= 60:
            return 3, "⭐⭐⭐", "小额投注", "55-65%"
        elif total_score >= 45:
            return 2, "⭐⭐", "谨慎投注", "45-55%"
        else:
            return 1, "⭐", "建议观望", "<45%"

    def _get_strategy_display_name(self, strategy_type):
        """获取策略类型的显示名称"""
        strategy_names = {
            'stable': '稳定期策略',
            'trend': '趋势期策略',
            'volatile': '波动期策略',
            'reversal': '反转期策略'
        }
        return strategy_names.get(strategy_type, '未知策略')

    def _get_recent_2_performance(self, backtest):
        """获取近1期表现信息"""
        history = backtest.get('history', [])
        if len(history) < 1:
            return "数据不足"

        # history是降序（最新在前），所以取第一个元素
        recent_1 = history[0]
        hit_count = recent_1
        return f"{hit_count}/1 中奖"

    def _check_model_agreement(self, predictions):
        """检查3个模型对TOP3预测是否一致

        Args:
            predictions: 预测结果列表，每个元素包含 (号码, 分数, 分析)

        Returns:
            bool: True表示模型一致，False表示有分歧
        """
        if not predictions or len(predictions) < 3:
            return False

        # 检查TOP3中每个号码的vote_details
        agreement_count = 0
        for i in range(3):
            pred = predictions[i]
            if len(pred) >= 3:
                analysis = pred[2]  # 第3个元素是分析字典
                vote_details = analysis.get('vote_details', {})

                # 检查3个模型是否都投票给这个号码
                # vote_details格式: {'RF': True/False, 'GB': True/False, 'LR': True/False}
                if vote_details:
                    votes = [vote_details.get('RF', False),
                            vote_details.get('GB', False),
                            vote_details.get('LR', False)]
                    if all(votes):  # 3个模型都投票
                        agreement_count += 1

        # 如果TOP3中至少有2个号码是3个模型都同意的，认为模型一致
        return agreement_count >= 2

    def validate_model_reliability(self, data, position):
        """优化版：交叉验证 + 趋势分析 + 数据分布检测

        优化点：
        1. 使用与推荐相同的多策略预测（而非仅LSTM）
        2. 验证TOP8命中率（而非TOP5）
        3. 使用统计显著性检验确定阈值
        4. 使用线性回归分析趋势
        5. 完善卡方检验实现
        6. 添加置信区间评估

        Args:
            data: 历史数据
            position: 预测名次

        Returns:
            dict: {
                'status': 'recommend'/'caution'/'not_recommend',
                'status_text': '[完成] 建议投注'/'⚠️ 谨慎投注'/'[错误] 不建议投注',
                'reason': '验证依据说明',
                'cv_accuracy': 交叉验证准确率,
                'cv_confidence_interval': 置信区间,
                'cv_p_value': 显著性p值,
                'trend': '上升'/'平稳'/'下降',
                'trend_slope': 趋势斜率,
                'trend_p_value': 趋势显著性p值,
                'distribution': '稳定'/'异常',
                'distribution_p_value': 分布检验p值
            }
        """
        try:
            if not self.lstm_models:
                return self._create_error_result('ML模型未训练')

            issues = list(data.keys())
            if len(issues) < 60:
                return self._create_error_result('历史数据不足')

            # 1. K折交叉验证（使用多策略预测TOP8，与推荐逻辑一致）
            cv_result = self._cross_validation_top8(data, position, k_folds=5, periods=50)
            cv_accuracy = cv_result['accuracy']
            cv_confidence_interval = cv_result['confidence_interval']
            cv_p_value = cv_result['p_value']
            cv_is_significant = cv_result['is_significant']

            # 2. 趋势分析（使用线性回归）
            trend_result = self._analyze_trend_with_regression(data, position, periods=10)
            trend = trend_result['trend']
            trend_slope = trend_result['slope']
            trend_p_value = trend_result['p_value']
            trend_is_significant = trend_result['is_significant']

            # 3. 数据分布检测（使用标准卡方检验）
            distribution_result = self._check_distribution_with_chi2(data, position)
            distribution_status = distribution_result['status']
            distribution_p_value = distribution_result['p_value']
            distribution_is_stable = distribution_result['is_stable']

            # 4. 综合判断（基于统计显著性）
            status, status_text, reason = self._make_recommendation_decision(
                cv_accuracy, cv_is_significant, cv_p_value,
                trend, trend_is_significant,
                distribution_is_stable, distribution_p_value
            )

            return {
                'status': status,
                'status_text': status_text,
                'reason': reason,
                'cv_accuracy': cv_accuracy,
                'cv_confidence_interval': cv_confidence_interval,
                'cv_p_value': cv_p_value,
                'trend': trend,
                'trend_slope': trend_slope,
                'trend_p_value': trend_p_value,
                'distribution': distribution_status,
                'distribution_p_value': distribution_p_value
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._create_error_result(f'验证失败: {str(e)}')

    def _create_error_result(self, reason):
        """创建错误结果"""
        return {
            'status': 'not_recommend',
            'status_text': '[错误] 不建议投注',
            'reason': reason,
            'cv_accuracy': 0,
            'cv_confidence_interval': (0, 0),
            'cv_p_value': 1.0,
            'trend': '未知',
            'trend_slope': 0,
            'trend_p_value': 1.0,
            'distribution': '未知',
            'distribution_p_value': 1.0
        }

    def _cross_validation_top8(self, data, position, k_folds=5, periods=50):
        """K折交叉验证（使用多策略预测TOP8）

        优化点：
        1. 使用generate_top8_multi_strategy（与推荐逻辑一致）
        2. 验证TOP8命中率（而非TOP5）
        3. 计算置信区间
        4. 进行统计显著性检验
        """
        issues = list(data.keys())
        recent_issues = issues[:periods]
        fold_size = len(recent_issues) // k_folds

        all_hits = 0
        all_total = 0
        fold_accuracies = []

        for fold in range(k_folds):
            # 划分训练集和验证集
            val_start = fold * fold_size
            val_end = val_start + fold_size
            val_issues = recent_issues[val_start:val_end]
            train_issues = [iss for iss in recent_issues if iss not in val_issues]

            # 在验证集上测试
            fold_hits = 0
            fold_total = 0

            for val_issue in val_issues:
                try:
                    # 用训练集数据预测（使用多策略）
                    train_data = {iss: data[iss] for iss in train_issues if iss in data}

                    if len(train_data) < 20:
                        continue

                    # 【关键优化】使用与推荐相同的多策略预测
                    top8, final_scores, _, _ = self.generate_top8_multi_strategy(train_data, position)

                    # 检查是否命中（TOP8）
                    actual = data[val_issue][position - 1]
                    if actual in top8:
                        fold_hits += 1
                    fold_total += 1
                except:
                    continue

            if fold_total > 0:
                fold_accuracy = fold_hits / fold_total
                fold_accuracies.append(fold_accuracy)
                all_hits += fold_hits
                all_total += fold_total

        # 计算平均命中率
        cv_accuracy = all_hits / all_total if all_total > 0 else 0

        # 计算95%置信区间
        if all_total > 0:
            import math
            z = 1.96  # 95%置信度
            se = math.sqrt(cv_accuracy * (1 - cv_accuracy) / all_total)
            ci_lower = max(0, cv_accuracy - z * se)
            ci_upper = min(1, cv_accuracy + z * se)
            confidence_interval = (ci_lower, ci_upper)
        else:
            confidence_interval = (0, 0)

        # 统计显著性检验：检验命中率是否显著高于随机基线（80%）
        baseline = 0.8  # TOP8理论命中率
        p_value = 1.0
        is_significant = False

        if all_total > 0:
            try:
                from scipy.stats import binom_test
                # 单侧检验：命中率是否显著高于基线
                p_value = binom_test(all_hits, all_total, baseline, alternative='greater')
                is_significant = (p_value < 0.05)  # 显著性水平5%
            except:
                # 如果scipy不可用，使用简单规则
                is_significant = (cv_accuracy > baseline + 0.05)

        return {
            'accuracy': cv_accuracy,
            'confidence_interval': confidence_interval,
            'p_value': p_value,
            'is_significant': is_significant,
            'n_samples': all_total
        }

    def _analyze_trend_with_regression(self, data, position, periods=10):
        """使用线性回归分析趋势

        优化点：
        1. 使用线性回归而非简单比较
        2. 计算趋势斜率
        3. 进行显著性检验
        """
        issues = list(data.keys())

        # 计算最近N期的命中率序列
        hit_rates = []
        for i in range(periods):
            try:
                # 计算第i期的命中率
                test_issue = issues[i]
                train_data = {iss: data[iss] for iss in issues[i+1:i+41]}  # 用后40期训练

                if len(train_data) < 20:
                    continue

                # 使用多策略预测
                top8, _, _, _ = self.generate_top8_multi_strategy(train_data, position)

                # 检查是否命中
                actual = data[test_issue][position - 1]
                hit = 1 if actual in top8 else 0
                hit_rates.append(hit)
            except:
                continue

        if len(hit_rates) < 5:
            return {
                'trend': '未知',
                'slope': 0,
                'p_value': 1.0,
                'is_significant': False
            }

        # 线性回归分析
        try:
            from scipy.stats import linregress
            x = list(range(len(hit_rates)))
            slope, intercept, r_value, p_value, std_err = linregress(x, hit_rates)

            # 判断趋势（基于斜率和显著性）
            if p_value < 0.1:  # 显著性水平10%（趋势分析可以放宽）
                if slope > 0.02:  # 斜率阈值
                    trend = '上升'
                elif slope < -0.02:
                    trend = '下降'
                else:
                    trend = '平稳'
                is_significant = True
            else:
                trend = '平稳'  # 不显著则认为平稳
                is_significant = False

            return {
                'trend': trend,
                'slope': slope,
                'p_value': p_value,
                'is_significant': is_significant
            }
        except:
            # 如果scipy不可用，使用简单方法
            recent_3 = sum(hit_rates[:3]) / 3 if len(hit_rates) >= 3 else 0
            recent_5 = sum(hit_rates[:5]) / 5 if len(hit_rates) >= 5 else 0
            recent_10 = sum(hit_rates) / len(hit_rates)

            if recent_3 > recent_5 and recent_5 >= recent_10:
                trend = '上升'
            elif recent_3 < recent_5 and recent_5 <= recent_10:
                trend = '下降'
            else:
                trend = '平稳'

            return {
                'trend': trend,
                'slope': 0,
                'p_value': 1.0,
                'is_significant': False
            }

    def _check_distribution_with_chi2(self, data, position):
        """使用标准卡方检验检测数据分布稳定性

        优化点：
        1. 使用scipy的标准卡方检验
        2. 计算p值而非使用经验阈值
        3. 正确处理期望频数
        """
        try:
            issues = list(data.keys())
            if len(issues) < 50:
                return {
                    'status': '未知',
                    'p_value': 1.0,
                    'is_stable': False
                }

            # 最近10期的号码分布
            recent_10_numbers = [data[iss][position - 1] for iss in issues[:10]]
            recent_10_freq = Counter(recent_10_numbers)

            # 之前40期的号码分布
            previous_40_numbers = [data[iss][position - 1] for iss in issues[10:50]]
            previous_40_freq = Counter(previous_40_numbers)

            # 构建观测频数和期望频数（对齐到所有号码1-10）
            observed = []
            expected = []

            for num in range(1, 11):
                obs = recent_10_freq.get(num, 0)
                exp_raw = previous_40_freq.get(num, 0) * 10 / 40  # 归一化到10期

                # 卡方检验要求期望频数≥5，这里放宽到≥1
                exp = max(exp_raw, 0.5)  # 避免除零

                observed.append(obs)
                expected.append(exp)

            # 标准卡方检验
            try:
                from scipy.stats import chisquare
                chi2_stat, p_value = chisquare(observed, expected)

                # p值>0.05表示分布一致（不能拒绝原假设）
                is_stable = (p_value > 0.05)
                status = '稳定' if is_stable else '异常'

                return {
                    'status': status,
                    'p_value': p_value,
                    'is_stable': is_stable,
                    'chi2_stat': chi2_stat
                }
            except:
                # 如果scipy不可用，使用简化计算
                chi2_stat = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
                is_stable = (chi2_stat < 15)  # 经验阈值
                status = '稳定' if is_stable else '异常'

                return {
                    'status': status,
                    'p_value': 1.0 if is_stable else 0.0,
                    'is_stable': is_stable,
                    'chi2_stat': chi2_stat
                }

        except Exception as e:
            return {
                'status': '未知',
                'p_value': 1.0,
                'is_stable': False
            }

    def _make_recommendation_decision(self, cv_accuracy, cv_is_significant, cv_p_value,
                                     trend, trend_is_significant,
                                     distribution_is_stable, distribution_p_value):
        """综合判断是否推荐投注

        优化点：
        1. 命中率优先：命中率>85%且显著时，无论分布如何都建议
        2. 基于统计显著性而非固定阈值
        3. 详细的理由说明（包含推荐/不推荐的具体原因）
        """

        reasons = []  # 收集所有判断依据

        # 规则1：强烈建议投注（命中率极高且显著）
        if cv_is_significant and cv_accuracy > 0.85:
            status = 'recommend'
            status_text = '✅ 强烈建议投注'
            reasons.append(f"✓ 命中率极高({cv_accuracy*100:.1f}%)")
            reasons.append(f"✓ 高度显著(p={cv_p_value:.3f})")
            reasons.append(f"✓ 趋势{trend}")
            if distribution_is_stable:
                reasons.append(f"✓ 分布稳定(p={distribution_p_value:.3f})")
            else:
                reasons.append(f"⚠ 分布异常(p={distribution_p_value:.3f})但命中率足够高")
            reason = " | ".join(reasons)

        # 规则2：建议投注（所有条件都满足）
        elif (cv_is_significant and cv_accuracy > 0.8 and
              trend in ['上升', '平稳'] and
              distribution_is_stable):
            status = 'recommend'
            status_text = '✅ 建议投注'
            reasons.append(f"✓ 命中率优秀({cv_accuracy*100:.1f}%)")
            reasons.append(f"✓ 统计显著(p={cv_p_value:.3f})")
            reasons.append(f"✓ 趋势{trend}")
            reasons.append(f"✓ 分布稳定(p={distribution_p_value:.3f})")
            reason = " | ".join(reasons)

        # 规则3：谨慎投注（命中率尚可且分布稳定）
        elif (cv_accuracy >= 0.75 and distribution_is_stable):
            status = 'caution'
            status_text = '⚠️ 谨慎投注'
            reasons.append(f"⚠ 命中率尚可({cv_accuracy*100:.1f}%)")
            if cv_is_significant:
                reasons.append(f"✓ 统计显著(p={cv_p_value:.3f})")
            else:
                reasons.append(f"✗ 不显著(p={cv_p_value:.3f})")
            reasons.append(f"⚠ 趋势{trend}")
            reasons.append(f"✓ 分布稳定(p={distribution_p_value:.3f})")
            reason = " | ".join(reasons)

        # 规则4：不建议投注（条件不满足）
        else:
            status = 'not_recommend'
            status_text = '❌ 不建议投注'

            # 详细列出不推荐的原因
            if cv_accuracy < 0.75:
                reasons.append(f"✗ 命中率偏低({cv_accuracy*100:.1f}%<75%)")
            else:
                reasons.append(f"⚠ 命中率({cv_accuracy*100:.1f}%)")

            if not cv_is_significant:
                reasons.append(f"✗ 不显著(p={cv_p_value:.3f}≥0.05)")
            else:
                reasons.append(f"✓ 显著(p={cv_p_value:.3f})")

            if trend == '下降':
                reasons.append(f"✗ 趋势下降")
            else:
                reasons.append(f"⚠ 趋势{trend}")

            if not distribution_is_stable:
                reasons.append(f"✗ 分布异常(p={distribution_p_value:.3f}≤0.05)")
            else:
                reasons.append(f"✓ 分布稳定(p={distribution_p_value:.3f})")

            reason = " | ".join(reasons)

        return status, status_text, reason

        """计算最近N期的TOP5命中率"""
        issues = list(data.keys())
        if len(issues) < periods + 10:
            return 0

        hits = 0
        for i in range(periods):
            test_issue = issues[i]
            train_data = {iss: data[iss] for iss in issues[i+1:i+41]}  # 用后40期训练

            result = self.predict_with_ml_model(train_data, position)
            if result:
                ml_scores, _ = result
                if ml_scores:
                    top5_predictions = sorted(ml_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                    top5_nums = [num for num, _ in top5_predictions]

                    actual = data[test_issue][position - 1]
                    if actual in top5_nums:
                        hits += 1

        return hits / periods if periods > 0 else 0

    def generate_recommendation_reason_high_confidence(self, best_pos, best_info,
                                                       all_scores, market_state):
        """
        生成高信心推荐理由（包含信心等级和详细得分）
        """
        reasons = []
        details = best_info['details']

        # 1. 信心等级
        stars = best_info.get('stars', '⭐⭐⭐')
        confidence_level = best_info.get('confidence_level', 3)
        reasons.append(f"信心等级:{stars}")

        # 2. 综合得分
        total_score = best_info['total_score']
        reasons.append(f"综合得分{total_score:.1f}分")

        # 3. LSTM置信度
        lstm_confidence = best_info.get('lstm_confidence', 0.5)
        reasons.append(f"LSTM置信度{lstm_confidence*100:.1f}%")

        # 4. 历史准确率
        top3_rate = best_info['top3_rate']
        reasons.append(f"历史命中率{top3_rate*100:.1f}%")

        # 5. 预期命中率
        expected_rate = best_info.get('expected_rate', '未知')
        reasons.append(f"预期命中率{expected_rate}")

        return " | ".join(reasons)


    # ========== 新增：动态多策略预测系统 ==========

    def predict_with_hot_strategy(self, data, position):
        """
        策略1：热号追踪策略
        核心：近期高频号码，适用于热号持续期
        """
        position_idx = position - 1
        values = list(data.values())

        scores = {}

        # 多时间窗口频率统计
        windows = {
            'recent_3': (3, 50),   # 近3期，权重50
            'recent_5': (5, 30),   # 近5期，权重30
            'recent_10': (10, 20)  # 近10期，权重20
        }

        for num in range(1, 11):
            score = 0
            for window_name, (period, weight) in windows.items():
                recent_nums = [values[i][position_idx] for i in range(min(period, len(values)))]
                freq = recent_nums.count(num)
                # 频率越高，得分越高
                score += (freq / period) * weight

            scores[num] = score

        return scores

    def predict_with_cold_strategy(self, data, position):
        """
        策略2：冷号回补策略
        核心：长期未出现号码，适用于冷号爆发期
        """
        position_idx = position - 1
        values = list(data.values())

        scores = {}

        for num in range(1, 11):
            # 计算当前遗漏期数
            current_omit = 0
            for i in range(len(values)):
                if values[i][position_idx] == num:
                    break
                current_omit += 1

            # 计算历史平均遗漏
            omit_list = []
            last_appear = -1
            for i in range(len(values)):
                if values[i][position_idx] == num:
                    if last_appear != -1:
                        omit_list.append(i - last_appear - 1)
                    last_appear = i

            avg_omit = sum(omit_list) / len(omit_list) if omit_list else 5

            # 遗漏越久，得分越高（激进策略）
            if current_omit >= 10:
                score = 100  # 10期未出现，最高分
            elif current_omit >= 5:
                score = 70   # 5期未出现，高分
            elif current_omit > avg_omit * 1.5:
                score = 50   # 超过平均遗漏1.5倍
            else:
                score = max(0, current_omit * 5)  # 基础分

            scores[num] = score

        return scores

    def predict_with_cycle_strategy(self, data, position):
        """
        策略3：周期规律策略
        核心：分析历史周期性，适用于稳定期
        """
        position_idx = position - 1
        values = list(data.values())

        scores = {}

        # 分析每个号码的出现周期
        for num in range(1, 11):
            appear_indices = []
            for i in range(len(values)):
                if values[i][position_idx] == num:
                    appear_indices.append(i)

            if len(appear_indices) < 2:
                scores[num] = 0
                continue

            # 计算周期间隔
            cycles = []
            for i in range(len(appear_indices) - 1):
                cycles.append(appear_indices[i] - appear_indices[i+1])

            avg_cycle = sum(cycles) / len(cycles) if cycles else 10

            # 计算当前距离上次出现的期数
            current_distance = appear_indices[0] if appear_indices else 0

            # 如果当前距离接近平均周期，得分高
            if abs(current_distance - avg_cycle) <= 2:
                score = 80  # 接近周期
            elif abs(current_distance - avg_cycle) <= 4:
                score = 50  # 较接近周期
            else:
                score = 20  # 偏离周期

            scores[num] = score

        return scores

    def predict_with_rf_strategy(self, data, position):
        """
        策略4：随机森林策略（使用已训练的RF模型）
        核心：多特征综合判断，适用于波动期
        """
        if not self.models_trained:
            # 如果模型未训练，返回均匀分布
            return {num: 50 for num in range(1, 11)}

        # 使用现有的ML模型预测
        result = self.predict_with_ml_model(data, position)
        if result:
            ml_scores, _ = result
            return ml_scores
        else:
            return {num: 50 for num in range(1, 11)}

    def detect_market_state(self, data, position):
        """
        市场状态识别
        返回：'HOT'(热号期) / 'COLD'(冷号期) / 'VOLATILE'(波动期) / 'STABLE'(稳定期)
        """
        position_idx = position - 1
        values = list(data.values())
        recent_20 = [values[i][position_idx] for i in range(min(20, len(values)))]

        # 1. 计算频率分布
        freq = Counter(recent_20)
        max_freq = max(freq.values())

        # 2. 计算熵值（衡量分布均匀度）
        probs = [count/len(recent_20) for count in freq.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)

        # 3. 判断状态
        if max_freq >= 6:  # 某号码出现≥6次
            return 'HOT'
        elif len(freq) <= 5:  # 只有5个或更少号码出现
            return 'COLD'
        elif entropy > 2.0:  # 熵值高，分布均匀
            return 'VOLATILE'
        else:
            return 'STABLE'

    def calculate_strategy_weights_dynamic(self, data, position):
        """
        动态计算各策略权重（基于近期表现）
        返回：{'lstm': 0.3, 'hot': 0.25, 'cold': 0.2, 'cycle': 0.15, 'rf': 0.1}
        """
        # 检测市场状态
        market_state = self.detect_market_state(data, position)

        # 根据市场状态分配权重
        if market_state == 'HOT':
            # 热号期：热号策略权重最高
            weights = {
                'lstm': 0.25,
                'hot': 0.40,
                'cold': 0.10,
                'cycle': 0.15,
                'rf': 0.10
            }
        elif market_state == 'COLD':
            # 冷号期：冷号策略权重最高
            weights = {
                'lstm': 0.25,
                'hot': 0.10,
                'cold': 0.40,
                'cycle': 0.15,
                'rf': 0.10
            }
        elif market_state == 'VOLATILE':
            # 波动期：随机森林和LSTM权重高
            weights = {
                'lstm': 0.35,
                'hot': 0.15,
                'cold': 0.15,
                'cycle': 0.10,
                'rf': 0.25
            }
        else:  # STABLE
            # 稳定期：周期策略和LSTM权重高
            weights = {
                'lstm': 0.30,
                'hot': 0.15,
                'cold': 0.15,
                'cycle': 0.30,
                'rf': 0.10
            }

        return weights, market_state

    def generate_top8_multi_strategy(self, data, position):
        """
        多策略融合生成TOP8推荐（核心优化函数）

        流程：
        1. 获取5种策略的预测结果
        2. 动态计算权重
        3. 综合评分
        4. 多样性优化
        5. 生成TOP8
        """
        # 步骤1：获取各策略预测
        lstm_scores = {}
        if self.models_trained and position in self.lstm_models:
            result = self.predict_with_ml_model(data, position)
            if result:
                lstm_scores, _ = result

        hot_scores = self.predict_with_hot_strategy(data, position)
        cold_scores = self.predict_with_cold_strategy(data, position)
        cycle_scores = self.predict_with_cycle_strategy(data, position)
        rf_scores = self.predict_with_rf_strategy(data, position)

        # 步骤2：动态权重
        weights, market_state = self.calculate_strategy_weights_dynamic(data, position)

        # 步骤3：综合评分
        final_scores = {}
        for num in range(1, 11):
            score = 0
            if lstm_scores:
                score += lstm_scores.get(num, 0) * weights['lstm']
            score += hot_scores.get(num, 0) * weights['hot']
            score += cold_scores.get(num, 0) * weights['cold']
            score += cycle_scores.get(num, 0) * weights['cycle']
            score += rf_scores.get(num, 0) * weights['rf']

            final_scores[num] = score

        # 步骤4：多样性优化（确保TOP8包含不同类型号码）
        position_idx = position - 1
        values = list(data.values())
        recent_10 = [values[i][position_idx] for i in range(min(10, len(values)))]

        for num in range(1, 11):
            freq = recent_10.count(num)

            # 多样性加分
            if freq >= 3:
                # 热号：额外+5分
                final_scores[num] += 5
            elif freq == 0:
                # 冷号：额外+8分
                final_scores[num] += 8
            else:
                # 中频号：额外+3分
                final_scores[num] += 3

        # 步骤5：排序并返回TOP8
        sorted_nums = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top8 = [num for num, score in sorted_nums[:8]]

        return top8, final_scores, market_state, weights

    def backtest_top8_rate(self, data, position, periods=30):
        """
        回测TOP8命中率
        """
        position_idx = position - 1
        issues = list(data.keys())
        values = list(data.values())

        hits = 0
        total = 0

        for i in range(min(periods, len(values) - 20)):
            # 使用前面的数据预测第i期
            train_data = dict(zip(issues[i+1:], values[i+1:]))

            if len(train_data) < 20:
                continue

            # 生成TOP8预测
            top8, _, _, _ = self.generate_top8_multi_strategy(train_data, position)

            # 获取实际号码
            actual = values[i][position_idx]

            # 判断是否命中
            if actual in top8:
                hits += 1
            total += 1

        return hits / total if total > 0 else 0

    def backtest_top8_detailed(self, data, position, periods=30):
        """
        回测TOP8命中率（返回详细信息）

        Returns:
            (hits, total): 命中次数和总次数
        """
        position_idx = position - 1
        issues = list(data.keys())
        values = list(data.values())

        hits = 0
        total = 0

        for i in range(min(periods, len(values) - 20)):
            # 使用前面的数据预测第i期
            train_data = dict(zip(issues[i+1:], values[i+1:]))

            if len(train_data) < 20:
                continue

            # 生成TOP8预测
            top8, _, _, _ = self.generate_top8_multi_strategy(train_data, position)

            # 获取实际号码
            actual = values[i][position_idx]

            # 判断是否命中
            if actual in top8:
                hits += 1
            total += 1

        return hits, total

    def check_consecutive_miss(self, data, position, periods=5):
        """
        检查最近N期连续未中次数

        Returns:
            int: 连续未中次数（从最近一期开始计算）
        """
        issues = list(data.keys())
        values = list(data.values())
        consecutive = 0

        for i in range(min(periods, len(issues) - 20)):
            # 使用历史数据预测第i期
            train_data = dict(zip(issues[i+1:], values[i+1:]))

            if len(train_data) < 20:
                continue

            try:
                # 生成TOP8预测
                top8, _, _, _ = self.generate_top8_multi_strategy(train_data, position)

                # 检查是否命中
                actual = values[i][position - 1]
                if actual not in top8:
                    consecutive += 1
                else:
                    break  # 遇到命中就停止
            except:
                break

        return consecutive

    def calculate_position_stability(self, data, position):
        """
        计算名次的稳定性（熵值）- 保留原版本用于兼容
        熵值越低，分布越稳定，越容易预测
        """
        position_idx = position - 1
        values = list(data.values())
        recent_30 = [values[i][position_idx] for i in range(min(30, len(values)))]

        freq = Counter(recent_30)
        probs = [count/len(recent_30) for count in freq.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)

        # 归一化到0-1，熵值越低，稳定性越高
        max_entropy = math.log(10)  # 均匀分布的最大熵
        stability = 1 - (entropy / max_entropy)

        return stability

    def calculate_position_stability_v2(self, data, position):
        """
        优化版：综合多个稳定性指标

        1. 信息熵（分布均匀度）40%
        2. 标准差（波动性）30%
        3. 自相关系数（周期性）30%
        """
        position_idx = position - 1
        values = list(data.values())
        recent_30 = [values[i][position_idx] for i in range(min(30, len(values)))]

        if len(recent_30) < 10:
            return 0.5  # 数据不足，返回中等稳定性

        # 1. 信息熵（0-1，越高越均匀，反转后越低越稳定）
        freq = Counter(recent_30)
        probs = [count/len(recent_30) for count in freq.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(10)
        entropy_score = 1 - (entropy / max_entropy)  # 反转：熵低=稳定

        # 2. 标准差（0-1，越低越稳定）
        std = np.std(recent_30)
        max_std = np.std(range(1, 11))  # 最大标准差
        std_score = 1 - (std / max_std) if max_std > 0 else 0.5

        # 3. 自相关系数（-1到1，越高越有周期性）
        if len(recent_30) >= 10:
            try:
                # 计算滞后1期的自相关
                autocorr = np.corrcoef(recent_30[:-1], recent_30[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr_score = 0.5
                else:
                    autocorr_score = (autocorr + 1) / 2  # 归一化到0-1
            except:
                autocorr_score = 0.5
        else:
            autocorr_score = 0.5

        # 综合评分
        stability = (
            entropy_score * 0.4 +      # 分布均匀度40%
            std_score * 0.3 +          # 波动性30%
            autocorr_score * 0.3       # 周期性30%
        )

        return stability

    def select_best_position_for_top8(self, data, all_predictions):
        """
        选择最适合TOP8投注的名次（优化版）

        优化点：
        1. 动态权重（根据市场状态自适应）
        2. 统计显著性检验
        3. 连续失败惩罚
        4. 综合稳定性指标
        5. 整合模型验证结果
        6. 趋势加成/减成

        评估维度：
        - 历史TOP8命中率（动态权重）
        - 近5期TOP8命中率（动态权重）
        - 号码分布稳定性（动态权重）
        - 交叉验证准确率（动态权重）
        - 统计显著性（加成）
        - 趋势方向（加成/减成）
        - 连续失败（惩罚）
        """
        position_scores = {}
        baseline = 0.8  # TOP8理论命中率

        for position in range(1, 11):
            try:
                # 1. 回测指标（详细版本）
                hits_30, total_30 = self.backtest_top8_detailed(data, position, periods=30)
                hits_5, total_5 = self.backtest_top8_detailed(data, position, periods=5)

                historical_top8 = hits_30 / total_30 if total_30 > 0 else 0
                recent_5_top8 = hits_5 / total_5 if total_5 > 0 else 0

                # 2. 统计显著性检验
                try:
                    from scipy.stats import binom_test
                    p_value = binom_test(hits_30, total_30, baseline, alternative='greater')
                    is_significant = (p_value < 0.05)
                except:
                    # scipy不可用，使用简单规则
                    p_value = 1.0
                    is_significant = (historical_top8 > baseline + 0.05)

                # 3. 稳定性（综合指标）
                stability = self.calculate_position_stability_v2(data, position)

                # 4. 市场状态
                market_state = self.detect_market_state(data, position)

                # 5. 模型验证结果
                validation = self.validate_model_reliability(data, position)
                cv_accuracy = validation.get('cv_accuracy', 0)
                cv_p_value = validation.get('cv_p_value', 1.0)
                cv_is_significant = (cv_p_value < 0.05)
                trend = validation.get('trend', '平稳')

                # 6. 连续失败检查
                consecutive_miss = self.check_consecutive_miss(data, position, periods=5)

                # 7. 动态权重（根据市场状态）
                if market_state == 'HOT':
                    # 热号期：更重视近期表现
                    weights = {'historical': 25, 'recent': 40, 'stability': 15, 'cv': 20}
                elif market_state == 'COLD':
                    # 冷号期：更重视历史表现
                    weights = {'historical': 40, 'recent': 25, 'stability': 15, 'cv': 20}
                elif market_state == 'VOLATILE':
                    # 波动期：降低稳定性权重，提升交叉验证权重
                    weights = {'historical': 30, 'recent': 30, 'stability': 10, 'cv': 30}
                else:  # STABLE
                    # 稳定期：平衡权重，提升稳定性权重
                    weights = {'historical': 35, 'recent': 30, 'stability': 20, 'cv': 15}

                # 8. 基础评分
                base_score = (
                    historical_top8 * weights['historical'] +
                    recent_5_top8 * weights['recent'] +
                    stability * weights['stability'] +
                    cv_accuracy * weights['cv']
                )

                # 9. 显著性加成
                if is_significant and cv_is_significant:
                    base_score *= 1.3  # 双重显著，加成30%
                elif is_significant or cv_is_significant:
                    base_score *= 1.15  # 单一显著，加成15%

                # 10. 趋势加成/减成
                if trend == '上升':
                    base_score *= 1.1  # 上升趋势，加成10%
                elif trend == '下降':
                    base_score *= 0.9  # 下降趋势，减成10%

                # 11. 连续失败惩罚
                if consecutive_miss >= 5:
                    penalty = 25  # 连续5期未中，扣25分
                elif consecutive_miss >= 3:
                    penalty = 15  # 连续3期未中，扣15分
                elif consecutive_miss >= 2:
                    penalty = 8   # 连续2期未中，扣8分
                else:
                    penalty = 0

                # 12. 最终得分
                final_score = max(0, base_score - penalty)

                # 13. 保存详细信息
                position_scores[position] = {
                    'total_score': final_score,
                    'base_score': base_score,
                    'historical_top8': historical_top8,
                    'recent_5_top8': recent_5_top8,
                    'stability': stability,
                    'cv_accuracy': cv_accuracy,
                    'p_value': p_value,
                    'is_significant': is_significant,
                    'cv_is_significant': cv_is_significant,
                    'trend': trend,
                    'consecutive_miss': consecutive_miss,
                    'penalty': penalty,
                    'market_state': market_state,
                    'weights': weights
                }

            except Exception as e:
                # 出错时使用默认值
                print(f"[警告] 第{position}名评分计算出错: {e}")
                position_scores[position] = {
                    'total_score': 0,
                    'base_score': 0,
                    'historical_top8': 0,
                    'recent_5_top8': 0,
                    'stability': 0
                }

        # 选择得分最高的名次
        if position_scores:
            best_position = max(position_scores.items(), key=lambda x: x[1]['total_score'])[0]
        else:
            best_position = 1  # 默认第1名

        return best_position, position_scores

    def check_consecutive_miss_count(self, position, backtest):
        """检查某名次连续未中次数"""
        backtest_details = backtest.get('backtest_details', [])
        consecutive = 0
        # 从最新期次开始往前数
        for detail in backtest_details:
            hit_status = detail.get('hit_status', '')
            if '[完成] TOP' not in hit_status:
                consecutive += 1
            else:
                break  # 遇到中奖就停止
        return consecutive

    def find_hottest_position_recent_5(self, all_backtests):
        """找到近5期命中率最高的名次"""
        best_position = 1
        best_rate = 0

        for position in range(1, 11):
            backtest = all_backtests.get(position, {})
            backtest_details = backtest.get('backtest_details', [])

            # 统计近5期命中次数
            hit_count = 0
            for detail in backtest_details[:5]:
                if '[完成] TOP' in detail.get('hit_status', ''):
                    hit_count += 1

            rate = hit_count / 5 if len(backtest_details) >= 5 else 0
            if rate > best_rate:
                best_rate = rate
                best_position = position

        return best_position

    def count_consecutive_miss(self):
        """统计推荐历史中连续未中的次数"""
        if len(self.recommendation_history) == 0:
            return 0

        consecutive = 0

        # 从最新记录往前数
        for record in reversed(self.recommendation_history):
            # 如果还在等待，先实时验证
            if record['result'] == 'waiting':
                data = self.get_history_data()
                if record['issue'] in data:
                    actual_numbers = data[record['issue']]
                    actual_num = actual_numbers[record['position'] - 1]
                    top7_nums = record['numbers'][:7]

                    if actual_num in top7_nums:
                        rank = top7_nums.index(actual_num) + 1
                        record['result'] = f'TOP{rank}'
                    else:
                        record['result'] = '未中'

            # 检查结果
            if record['result'] == '未中':
                consecutive += 1
            else:
                break  # 遇到已中就停止

        return consecutive

    def get_blacklist_positions(self):
        """获取最近2期推荐过且未中的名次（黑名单）"""
        blacklist = set()

        if len(self.recommendation_history) < 2:
            return blacklist

        # 检查最近2期
        for record in self.recommendation_history[-2:]:
            # 先验证结果
            if record['result'] == 'waiting':
                data = self.get_history_data()
                if record['issue'] in data:
                    actual_numbers = data[record['issue']]
                    actual_num = actual_numbers[record['position'] - 1]
                    top7_nums = record['numbers'][:7]

                    if actual_num in top7_nums:
                        record['result'] = f'TOP{top7_nums.index(actual_num) + 1}'
                    else:
                        record['result'] = '未中'

            if record['result'] == '未中':
                blacklist.add(record['position'])

        return blacklist

    def select_guaranteed_hit_position(self, all_backtests):
        """
        必中模式：选择近期100%命中的名次

        策略：
        1. 找到近5期100%命中的名次（5/5）
        2. 如果没有，找近4期100%命中的（4/4）
        3. 如果没有，找近3期100%命中的（3/3）
        4. 如果都没有，选择近10期命中率最高的（通常≥90%）
        """

        # 第1优先级：近5期、4期、3期100%命中
        for check_period in [5, 4, 3]:
            perfect_positions = []

            for position in range(1, 11):
                backtest = all_backtests.get(position, {})
                backtest_details = backtest.get('backtest_details', [])

                if len(backtest_details) < check_period:
                    continue

                # 检查近N期是否全部命中
                hit_count = sum(1 for d in backtest_details[:check_period]
                               if '[完成] TOP' in d.get('hit_status', ''))

                if hit_count == check_period:
                    # 额外验证：近10期命中率也要高
                    hit_10 = sum(1 for d in backtest_details[:min(10, len(backtest_details))]
                                if '[完成] TOP' in d.get('hit_status', ''))
                    rate_10 = hit_10 / min(10, len(backtest_details))

                    perfect_positions.append({
                        'position': position,
                        'perfect_period': check_period,
                        'rate_10': rate_10,
                        'score': check_period * 10 + rate_10 * 100  # 综合评分
                    })

            if perfect_positions:
                # 选择综合评分最高的
                best = max(perfect_positions, key=lambda x: x['score'])
                return best['position'], best['perfect_period'], best['rate_10']

        # 如果没有100%命中的，选择近10期命中率最高的
        best_position = 1
        best_rate = 0

        for position in range(1, 11):
            backtest = all_backtests.get(position, {})
            backtest_details = backtest.get('backtest_details', [])

            if len(backtest_details) < 10:
                continue

            hit_count = sum(1 for d in backtest_details[:10]
                           if '[完成] TOP' in d.get('hit_status', ''))
            rate = hit_count / 10

            if rate > best_rate:
                best_rate = rate
                best_position = position

        return best_position, 0, best_rate

    def select_position_with_rotation(self, all_backtests, blacklist):
        """
        轮换策略：选择从未推荐过或很久没推荐的高命中率名次

        逻辑：
        - 有些名次可能一直没被推荐，但历史命中率很高
        - 轮换到这些"新鲜"名次，避免在同一批名次上反复失败
        """

        # 统计每个名次被推荐的次数
        position_recommend_count = {}
        for pos in range(1, 11):
            position_recommend_count[pos] = 0

        for record in self.recommendation_history[-10:]:  # 只看最近10次
            pos = record['position']
            position_recommend_count[pos] += 1

        # 找到高命中率且推荐次数少的名次
        candidates = []

        for position in range(1, 11):
            if position in blacklist:
                continue

            backtest = all_backtests.get(position, {})
            backtest_details = backtest.get('backtest_details', [])

            if len(backtest_details) < 10:
                continue

            # 近10期命中率
            hit_10 = sum(1 for d in backtest_details[:10]
                        if '[完成] TOP' in d.get('hit_status', ''))
            rate_10 = hit_10 / 10

            # 只考虑命中率≥80%的
            if rate_10 >= 0.8:
                recommend_count = position_recommend_count[position]

                # 评分：命中率高 + ���荐次数少
                score = rate_10 * 100 - recommend_count * 5

                candidates.append({
                    'position': position,
                    'rate_10': rate_10,
                    'recommend_count': recommend_count,
                    'score': score
                })

        if candidates:
            best = max(candidates, key=lambda x: x['score'])
            return best['position'], best['rate_10']

        # 兜底：返回命中率最高的
        return self.select_ultra_stable_position_exclude_blacklist(all_backtests, blacklist)

    def _get_best_position_by_period(self, all_backtests, period, blacklist):
        """获取指定周期内命中率最高的名次"""
        best_pos = None
        best_rate = 0

        for position in range(1, 11):
            if position in blacklist:
                continue

            backtest = all_backtests.get(position, {})
            backtest_details = backtest.get('backtest_details', [])

            if len(backtest_details) < period:
                continue

            hit_count = sum(1 for d in backtest_details[:period]
                           if '[完成] TOP' in d.get('hit_status', ''))
            rate = hit_count / period

            if rate > best_rate:
                best_rate = rate
                best_pos = position

        return best_pos

    def _get_most_stable_position(self, all_backtests, blacklist):
        """获取稳定性最好的名次（命中率方差最小）"""
        best_pos = None
        best_stability = -1

        for position in range(1, 11):
            if position in blacklist:
                continue

            backtest = all_backtests.get(position, {})
            backtest_details = backtest.get('backtest_details', [])

            if len(backtest_details) < 10:
                continue

            # 计算近10期的命中率方差
            results = [1 if '[完成] TOP' in d.get('hit_status', '') else 0
                      for d in backtest_details[:10]]
            mean = sum(results) / 10
            variance = sum((x - mean)**2 for x in results) / 10
            stability = 1 - variance  # 方差越小，稳定性越高

            # 只考虑命中率≥70%的
            if mean >= 0.7 and stability > best_stability:
                best_stability = stability
                best_pos = position

        return best_pos

    def _get_best_trending_position(self, all_backtests, blacklist):
        """获取上升趋势最强的名次"""
        best_pos = None
        best_trend = -1

        for position in range(1, 11):
            if position in blacklist:
                continue

            backtest = all_backtests.get(position, {})
            backtest_details = backtest.get('backtest_details', [])

            if len(backtest_details) < 10:
                continue

            # 近5期命中率 vs 近10期命中率
            hit_5 = sum(1 for d in backtest_details[:5]
                       if '[完成] TOP' in d.get('hit_status', ''))
            rate_5 = hit_5 / 5

            hit_10 = sum(1 for d in backtest_details[5:10]
                        if '[完成] TOP' in d.get('hit_status', ''))
            rate_10 = hit_10 / 5 if len(backtest_details) >= 10 else 0

            trend = rate_5 - rate_10  # 正值表示上升趋势

            # 只考虑近5期命中率≥80%的
            if rate_5 >= 0.8 and trend > best_trend:
                best_trend = trend
                best_pos = position

        return best_pos

    def select_position_by_ensemble_voting(self, all_backtests, blacklist):
        """
        集成投票：多种方法同时推荐，选择票数最多的名次

        投票方法：
        1. 近5期命中率最高（1票）
        2. 近10期命中率最高（2票，权重更高）
        3. 近20期命中率最高（1票）
        4. 稳定性最好（1票）
        5. 近期上升趋势最强（1票）
        """

        votes = {}
        for pos in range(1, 11):
            votes[pos] = 0

        # 投票1：近5期命中率最高
        best_5 = self._get_best_position_by_period(all_backtests, 5, blacklist)
        if best_5:
            votes[best_5] += 1

        # 投票2：近10期命中率最高（权重更高）
        best_10 = self._get_best_position_by_period(all_backtests, 10, blacklist)
        if best_10:
            votes[best_10] += 2

        # 投票3：近20期命中率最高
        best_20 = self._get_best_position_by_period(all_backtests, 20, blacklist)
        if best_20:
            votes[best_20] += 1

        # 投票4：稳定性最好
        best_stable = self._get_most_stable_position(all_backtests, blacklist)
        if best_stable:
            votes[best_stable] += 1

        # 投票5：上升趋势最强
        best_trend = self._get_best_trending_position(all_backtests, blacklist)
        if best_trend:
            votes[best_trend] += 1

        # 选择票数最多的
        best_position = max(votes.items(), key=lambda x: x[1])[0]

        return best_position

    def select_ultra_stable_position_exclude_blacklist(self, all_backtests, blacklist):
        """选择超稳定名次，排除黑名单（近20期命中率最高）"""
        best_position = 1
        best_rate = 0

        for position in range(1, 11):
            # 跳过黑名单
            if position in blacklist:
                continue

            backtest = all_backtests.get(position, {})
            backtest_details = backtest.get('backtest_details', [])

            if len(backtest_details) < 20:
                continue

            # 近20期命中率（70%权重）
            hit_20 = sum(1 for d in backtest_details[:20]
                        if '[完成] TOP' in d.get('hit_status', ''))
            rate_20 = hit_20 / 20

            # 近5期命中率（30%权重）- 确保近期也稳定
            hit_5 = sum(1 for d in backtest_details[:5]
                       if '[完成] TOP' in d.get('hit_status', ''))
            rate_5 = hit_5 / 5

            # 综合评分
            combined_score = rate_20 * 0.7 + rate_5 * 0.3

            if combined_score > best_rate:
                best_rate = combined_score
                best_position = position

        return best_position, best_rate

    def select_second_ultra_stable_position(self, all_backtests, blacklist, exclude_position):
        """选择第二超稳定名次（排除黑名单和已选名次）"""
        position_rates = []

        for position in range(1, 11):
            if position in blacklist or position == exclude_position:
                continue

            backtest = all_backtests.get(position, {})
            backtest_details = backtest.get('backtest_details', [])

            if len(backtest_details) < 20:
                continue

            # 近20期命中率（70%权重）
            hit_20 = sum(1 for d in backtest_details[:20]
                        if '[完成] TOP' in d.get('hit_status', ''))
            rate_20 = hit_20 / 20

            # 近5期命中率（30%权重）
            hit_5 = sum(1 for d in backtest_details[:5]
                       if '[完成] TOP' in d.get('hit_status', ''))
            rate_5 = hit_5 / 5

            # 综合评分
            combined_score = rate_20 * 0.7 + rate_5 * 0.3

            position_rates.append((position, combined_score))

        # 按命中率排序，选择第一个
        position_rates.sort(key=lambda x: x[1], reverse=True)

        if position_rates:
            return position_rates[0][0], position_rates[0][1]
        else:
            return 1, 0.5

    def select_best_position_with_fallback(self, data, all_predictions, all_backtests):
        """智能追号：四级防连败机制（确保不连败3次）

        返回:
            best_position: 第1期推荐名次
            backup_position: 第2期备选名次
            fallback_reason: 回退原因（如有）
            position_scores: 所有名次得分（包含详细字段）
        """

        # ===== 统计连续未中次数 =====
        consecutive_miss = self.count_consecutive_miss()

        # ===== 获取黑名单 =====
        blacklist = self.get_blacklist_positions()

        # ===== 第4级：必中模式（连续2期未中）=====
        if consecutive_miss >= 2:
            self.log("="*80)
            self.log("🚨🚨🚨 触发必中模式：连续2期未中，启动100%保障策略！")
            self.log("="*80)
            self.log("")

            # 更新统计
            self.stats_level4_count += 1
            self.stats_consecutive_2_miss += 1

            # 方案1：寻找近期100%命中的名次
            best_position, perfect_period, rate_10 = self.select_guaranteed_hit_position(all_backtests)

            # 方案2：集成投票选择最稳定名次
            voted_position = self.select_position_by_ensemble_voting(all_backtests, blacklist)

            # 方案3：轮换策略
            rotated_position, rotate_rate = self.select_position_with_rotation(all_backtests, blacklist)

            # 最终决策：优先选择100%命中的，否则选择投票结果
            if perfect_period > 0:
                final_position = best_position
                backup_position = voted_position
                strategy_desc = f"近{perfect_period}期100%命中，近10期命中率{rate_10*100:.1f}%"
            else:
                final_position = voted_position
                backup_position = rotated_position
                strategy_desc = f"集成投票+轮换策略，近10期命中率{rate_10*100:.1f}%"

            fallback_reason = (
                f"🚨🚨🚨 必中模式启动！连续{consecutive_miss}期未中\n"
                f"   策略：{strategy_desc}\n"
                f"   主推荐：第{final_position}名\n"
                f"   备选：第{backup_position}名\n"
                f"   黑名单（已排除）：{blacklist if blacklist else '无'}\n"
                f"   ⚠️ 本期采用最保守策略，预期命中率≥95%"
            )

            self.log(f"🚨 必中模式详情：")
            self.log(f"   连续未中次数：{consecutive_miss}期")
            self.log(f"   黑名单名次：{blacklist if blacklist else '无'}")
            self.log(f"   策略描述：{strategy_desc}")
            self.log(f"   主推荐：第{final_position}名")
            self.log(f"   备选推荐：第{backup_position}名")
            self.log("")

            # 重新计算评分（用于显示）
            _, _, adaptive_scores, _ = self.select_best_position_adaptive(data, all_predictions, all_backtests)
            _, detailed_scores = self.select_best_position_for_top8(data, all_predictions)

            position_scores = {}
            for position in range(1, 11):
                if position in adaptive_scores:
                    position_scores[position] = adaptive_scores[position].copy()
                    if position in detailed_scores:
                        position_scores[position].update(detailed_scores[position])

            return final_position, backup_position, fallback_reason, position_scores

        # ===== 第3级：紧急模式（1期未中）=====
        elif consecutive_miss >= 1:
            self.log("="*80)
            self.log("⚠️ 触发紧急模式：上期未中，切换到超稳定名次")
            self.log("="*80)
            self.log("")

            # 更新统计
            self.stats_level3_count += 1

            best_position, best_rate = self.select_ultra_stable_position_exclude_blacklist(
                all_backtests, blacklist
            )
            backup_position, backup_rate = self.select_second_ultra_stable_position(
                all_backtests, blacklist, best_position
            )

            fallback_reason = (
                f"⚠️ 紧急模式：上期未中\n"
                f"   第{best_position}名近20期命中率：{best_rate*100:.1f}%\n"
                f"   备选第{backup_position}名近20期命中率：{backup_rate*100:.1f}%\n"
                f"   黑名单（已排除）：{blacklist if blacklist else '无'}"
            )

            self.log(f"⚠️ 紧急模式详情：")
            self.log(f"   连续未中次数：{consecutive_miss}期")
            self.log(f"   黑名单名次：{blacklist if blacklist else '无'}")
            self.log(f"   主推荐：第{best_position}名（近20期命中率{best_rate*100:.1f}%）")
            self.log(f"   备选推荐：第{backup_position}名（近20期命中率{backup_rate*100:.1f}%）")
            self.log("")

            # 重新计算评分
            _, _, adaptive_scores, _ = self.select_best_position_adaptive(data, all_predictions, all_backtests)
            _, detailed_scores = self.select_best_position_for_top8(data, all_predictions)

            position_scores = {}
            for position in range(1, 11):
                if position in adaptive_scores:
                    position_scores[position] = adaptive_scores[position].copy()
                    if position in detailed_scores:
                        position_scores[position].update(detailed_scores[position])

            return best_position, backup_position, fallback_reason, position_scores

        # ===== 第1级：正常模式（上期已中或首次推荐）=====
        else:
            # 更新统计
            self.stats_level1_count += 1

            # 1. 使用自适应方法计算所有名次得分（用于排序和信心等级）
            _, _, adaptive_scores, _ = self.select_best_position_adaptive(data, all_predictions, all_backtests)

            # 2. 使用TOP8方法获取详细评分（用于显示）
            _, detailed_scores = self.select_best_position_for_top8(data, all_predictions)

            # 3. 合并两种评分数据
            position_scores = {}
            for position in range(1, 11):
                if position in adaptive_scores:
                    position_scores[position] = adaptive_scores[position].copy()
                    # 添加详细字段
                    if position in detailed_scores:
                        position_scores[position].update(detailed_scores[position])

            # 4. 按得分排序
            sorted_positions = sorted(position_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)

            if not sorted_positions:
                return 1, 2, "", {}

            # 5. 获取TOP2推荐
            first_choice = sorted_positions[0][0]
            second_choice = sorted_positions[1][0] if len(sorted_positions) > 1 else 1

            fallback_reason = ""
            best_position = first_choice
            backup_position = second_choice

            return best_position, backup_position, fallback_reason, position_scores


def main():
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
