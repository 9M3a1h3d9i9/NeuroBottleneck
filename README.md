# NeuroBottleneck

> **پلتفرم پژوهشی برای هوشمندسازی، بهینه‌سازی و افزایش تاب‌آوری شبکه‌های ارتباطی با استفاده از یادگیری گراف و یادگیری تقویتی عمیق**
>
> **A research-oriented platform for intelligent optimization and resilience of communication networks using graph learning and deep reinforcement learning.**

---

## 📡 چشم‌انداز پروژه | Project Vision

**NeuroBottleneck** یک پروژه پژوهشی و مهندسی برای مطالعه‌ی هوشمندسازی شبکه‌های ارتباطی مدرن است که بر ترکیب **Graph Neural Networks (GNNs)**، تحلیل گراف، بهینه‌سازی شبکه و **Deep Reinforcement Learning (DRL)** تمرکز دارد.

**NeuroBottleneck** is a research and engineering project focused on intelligent management of modern communication networks through the integration of **Graph Neural Networks (GNNs)**, graph analytics, network optimization, and **Deep Reinforcement Learning (DRL)**.

هدف اصلی پروژه، ایجاد یک چارچوب قابل‌آزمایش و قابل‌تکرار برای شناسایی گلوگاه‌های شبکه، تحلیل بحرانی بودن اجزای شبکه و در مراحل بعد، انتخاب اقدامات بهینه برای کاهش ازدحام و افزایش تاب‌آوری شبکه است.

The primary objective is to establish an experimentally testable and reproducible framework for identifying network bottlenecks, analyzing component criticality, and, in later stages, selecting optimization actions that reduce congestion and improve network resilience.

---

## 🎯 مسئله پژوهشی | Research Problem

شبکه‌های ارتباطی پویا باید به‌صورت هم‌زمان میان **ظرفیت، تقاضای ترافیک، توپولوژی، کیفیت سرویس و تاب‌آوری** تعادل برقرار کنند. تغییرات مداوم در بار ترافیکی یا وضعیت اجزای شبکه می‌تواند باعث ایجاد نقاط بحرانی و گلوگاه‌هایی شود که عملکرد کل شبکه را تحت تأثیر قرار می‌دهند.

Dynamic communication networks must continuously balance **capacity, traffic demand, topology, quality of service, and resilience**. Changes in traffic load or network conditions can create critical components and bottlenecks that significantly affect overall network performance.

NeuroBottleneck این مسئله را به‌عنوان یک حلقه‌ی تصمیم‌گیری بسته مدل می‌کند که در آن وضعیت شبکه به نمایش گرافی تبدیل شده، نقاط بحرانی شناسایی شده و در نهایت عامل هوشمند برای انتخاب اقدام مناسب مورد استفاده قرار می‌گیرد.

NeuroBottleneck formulates this problem as a closed decision loop in which the network state is represented as a graph, critical components are identified, and an intelligent agent is eventually used to select appropriate optimization actions.

```text
Network State
      ↓
Graph Construction / Representation
      ↓
Network & Bottleneck Analysis
      ↓
State Representation
      ↓
Decision / RL Agent
      ↓
Optimization Action
      ↓
Network KPI Evaluation
      ↺
```

---

## 🧠 رویکرد علمی | Scientific Approach

پروژه بر یک معماری مرحله‌ای استوار است تا هر بخش از سیستم ابتدا به‌صورت مستقل قابل ارزیابی باشد و سپس در یک چارچوب یکپارچه قرار گیرد.

The project follows a staged architecture in which each component is independently testable before being integrated into the complete intelligent optimization framework.

در سطح پایه، شبکه به‌صورت یک **Graph G=(V,E)** مدل می‌شود که در آن گره‌ها می‌توانند نماینده‌ی عناصر شبکه و یال‌ها نماینده‌ی ارتباطات یا مسیرهای ارتباطی باشند.

At the fundamental level, the network is modeled as a **graph G=(V,E)**, where nodes may represent network entities and edges represent communication links or connectivity relationships.

ویژگی‌های گره‌ها و یال‌ها می‌توانند شامل ظرفیت، بار، تأخیر، نرخ ترافیک، وضعیت لینک، میزان استفاده از منابع و سایر شاخص‌های قابل‌اندازه‌گیری شبکه باشند.

Node and edge attributes may include capacity, utilization, latency, traffic rate, link state, resource usage, and other measurable network indicators.

در مراحل پیشرفته‌تر، این نمایش گرافی به ورودی مدل‌های یادگیری گراف و عامل یادگیری تقویتی تبدیل خواهد شد.

In later stages, this graph representation will serve as the input to graph-learning models and the reinforcement-learning agent.

---

## 🔬 اهداف پژوهشی | Research Objectives

### 1. مدل‌سازی گرافی شبکه | Graph-Based Network Modeling

ایجاد یک نمایش ساختاریافته از توپولوژی شبکه و وضعیت عملیاتی آن به‌گونه‌ای که تغییرات توپولوژی و ترافیک قابل بازنمایی و تحلیل باشند.

Develop a structured representation of network topology and operational state that can capture and analyze changes in topology and traffic conditions.

### 2. شناسایی گلوگاه‌ها | Bottleneck Identification

طراحی و ارزیابی روش‌های baseline برای شناسایی گره‌ها، لینک‌ها یا نواحی بحرانی شبکه با استفاده از شاخص‌های توپولوژیکی، ظرفیت و وضعیت ترافیکی.

Design and evaluate baseline methods for identifying critical nodes, links, or network regions using topological, capacity-related, and traffic-related indicators.

### 3. استخراج نمایش وضعیت شبکه | Network State Representation

بررسی روش‌های تبدیل وضعیت شبکه به یک نمایش مناسب برای الگوریتم‌های یادگیری ماشین و یادگیری تقویتی.

Investigate methods for transforming network states into representations suitable for machine learning and reinforcement learning algorithms.

### 4. بهینه‌سازی تطبیقی | Adaptive Optimization

در مراحل بعدی، بررسی این موضوع که آیا یک عامل DRL می‌تواند بر اساس وضعیت جاری شبکه، اقدام مناسبی برای کاهش گلوگاه و بهبود شاخص‌های عملکرد انتخاب کند.

In later stages, investigate whether a DRL agent can select appropriate actions based on the current network state to mitigate bottlenecks and improve network performance indicators.

### 5. افزایش تاب‌آوری | Network Resilience

بررسی تأثیر تصمیم‌های بهینه‌سازی بر توانایی شبکه برای حفظ عملکرد در شرایطی مانند افزایش بار، اختلال لینک یا تغییرات توپولوژیکی.

Investigate how optimization decisions affect the ability of the network to maintain performance under conditions such as increased traffic load, link failures, or topology changes.

### 6. تعمیم‌پذیری | Generalization

ارزیابی مدل‌ها روی سناریوها و توپولوژی‌هایی متفاوت از شرایط مورد استفاده در آموزش، به‌منظور بررسی میزان تعمیم‌پذیری روش پیشنهادی.

Evaluate the models on network scenarios and topologies different from those used during training to assess the generalization capability of the proposed approach.

---

# 🏗️ معماری مفهومی | Conceptual Architecture

```text
                ┌──────────────────────┐
                │   Network Scenario   │
                │ Traffic / Topology   │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ Graph Representation  │
                │      G = (V, E)      │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ Network Analytics     │
                │ Centrality / Capacity │
                │ Traffic / Criticality │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ State Representation  │
                │   GNN / Features      │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │    DRL Agent          │
                │ Policy / Value Model  │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ Optimization Action   │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ Network KPI Evaluation│
                └──────────┬───────────┘
                           │
                           └──────────────↺
```

این معماری به‌صورت عمدی **ماژولار** طراحی شده است تا امکان جایگزینی الگوریتم‌های مختلف بدون بازطراحی کل سامانه وجود داشته باشد.

This architecture is intentionally **modular**, allowing individual algorithms or components to be replaced without redesigning the entire system.

---

# 🚧 وضعیت فعلی پروژه | Current Development Status

> **Current Stage: Phase 2 — Environment & Baseline Modeling (Mid-Phase)**

پروژه در حال حاضر در **میانه‌ی فاز دوم توسعه** قرار دارد. فاز اول بر ایجاد پایه‌ی پژوهشی، تعریف مسئله، بررسی منابع، طراحی معماری و ایجاد ساختار اولیه‌ی نرم‌افزار متمرکز بود.

The project is currently in the **middle of Phase 2**. Phase 1 focused on establishing the research foundation, defining the problem, reviewing relevant literature and standards, designing the architecture, and creating the initial software structure.

در فاز دوم، تمرکز از طراحی مفهومی به سمت **ساخت محیط آزمایش، مدل‌سازی شبکه، تعریف KPIها و ایجاد baselineهای قابل‌اندازه‌گیری** منتقل شده است.

During Phase 2, the focus has shifted from conceptual design toward **experimental environment development, network modeling, KPI definition, and measurable baseline implementations**.

در این مرحله هنوز ادعایی درباره‌ی برتری روش پیشنهادی، بهبود درصدی عملکرد یا دستیابی به نتایج صنعتی مطرح نمی‌شود؛ این موارد تنها پس از اجرای آزمایش‌های کنترل‌شده و قابل‌تکرار گزارش خواهند شد.

At this stage, no claims are made regarding superiority, percentage performance improvements, or industrial-level results. Such claims will only be reported after controlled and reproducible experiments have been executed.

---

# 🗺️ نقشه راه | Roadmap

## Phase 1 — Research Foundation ✅

**فاز اول با هدف ایجاد پایه‌ی علمی و مهندسی پروژه انجام شده است.**

**Phase 1 established the scientific and engineering foundation of the project.**

- مرور ادبیات پژوهش
- Literature review
- بررسی مفاهیم GNN و DRL
- Review of GNN and DRL concepts
- مطالعه‌ی معماری‌های شبکه و O-RAN
- Study of networking and O-RAN architectures
- تعریف مسئله‌ی پژوهشی
- Research problem formulation
- طراحی معماری اولیه
- Initial architecture design
- ایجاد ساختار ماژولار مخزن
- Modular repository foundation
- ایجاد چارچوب مستندسازی و آزمایش
- Documentation and experimentation framework

---

# Phase 2 — Environment & Baseline Modeling 🚧

**وضعیت فعلی پروژه در این فاز: در حال توسعه — Mid-Phase.**

**Current project status in this phase: under active development — Mid-Phase.**

### اهداف اصلی این فاز | Main Objectives

- ایجاد محیط آزمایش شبکه
- Develop the network experimentation environment
- تعریف ساختار داده‌ی گراف
- Define graph data structures
- تعریف ویژگی‌های Node و Edge
- Define node and edge attributes
- مدل‌سازی ظرفیت و بار شبکه
- Model network capacity and utilization
- تعریف شاخص‌های کلیدی عملکرد
- Define measurable network KPIs
- پیاده‌سازی روش‌های baseline برای bottleneck detection
- Implement baseline bottleneck-detection methods
- ایجاد سناریوهای ترافیکی و اختلال
- Develop traffic and failure scenarios
- آماده‌سازی محیط RL
- Prepare the reinforcement-learning environment
- ایجاد baseline policyها برای مقایسه‌ی آینده
- Establish baseline policies for future comparison

### KPIهای موردنظر | Target Evaluation Metrics

ارزیابی آینده‌ی سیستم بر اساس شاخص‌هایی مانند موارد زیر انجام خواهد شد:

Future evaluation will consider measurable indicators such as:

| شاخص | Metric |
|---|---|
| میانگین تأخیر | Mean Latency |
| نرخ از دست رفتن بسته | Packet Loss Rate |
| توان عملیاتی | Throughput |
| میزان استفاده از ظرفیت | Resource / Link Utilization |
| تعداد یا شدت گلوگاه‌ها | Bottleneck Count / Severity |
| زمان تصمیم‌گیری | Decision Latency |
| پایداری عملکرد تحت بار | Performance Stability under Load |
| عملکرد تحت خرابی | Failure Resilience |
| عملکرد روی توپولوژی‌های دیده‌نشده | Generalization to Unseen Topologies |

> **مقادیر عددی این شاخص‌ها تنها پس از اجرای آزمایش و ثبت artifactهای مربوطه در repository گزارش خواهند شد.**

> **Numerical values for these metrics will only be reported after experiments are executed and the corresponding artifacts are stored in the repository.**

---

# Phase 3 — Intelligent Optimization 🔬

**در این فاز، مدل‌های یادگیری گراف و یادگیری تقویتی عمیق در حلقه‌ی تصمیم‌گیری قرار خواهند گرفت.**

**In this phase, graph-learning models and deep reinforcement learning will be introduced into the decision-making loop.**

اهداف اصلی شامل موارد زیر خواهد بود:

The main objectives will include:

- GNN-based state representation
- نمایش وضعیت شبکه با GNN
- طراحی فضای State و Action
- State and action-space design
- طراحی تابع Reward
- Reward-function design
- پیاده‌سازی عامل DRL
- DRL agent implementation
- مدل‌سازی Dynamic Traffic
- Dynamic traffic modeling
- سناریوهای Link Failure و Network Perturbation
- Link-failure and network-perturbation scenarios
- مقایسه با baselineها
- Comparison against established baselines
- تحلیل حساسیت و Ablation Study
- Sensitivity analysis and ablation studies

---

# Phase 4 — Robustness & Research Evaluation 📊

**فاز چهارم بر ارزیابی علمی، robustness و generalization متمرکز خواهد بود.**

**Phase 4 will focus on scientific evaluation, robustness, and generalization.**

ارزیابی نهایی باید حداقل شامل مقایسه با baselineهای مشخص، چند سناریوی ترافیکی، چند وضعیت توپولوژیکی و آزمایش‌های ablation باشد.

The final evaluation should include comparison against defined baselines, multiple traffic scenarios, multiple topology conditions, and controlled ablation studies.

تمرکز این فاز بر پاسخ به این پرسش خواهد بود که آیا روش پیشنهادی تنها در یک سناریوی خاص عملکرد مناسبی دارد یا می‌تواند در شرایط متفاوت نیز رفتار پایدار و قابل‌تعمیمی ارائه کند.

The central question in this phase is whether the proposed approach performs well only under a specific scenario or can maintain stable and generalizable behavior under different network conditions.

---

# 🧪 معیارهای پژوهش قابل‌تکرار | Reproducible Research Criteria

NeuroBottleneck تلاش می‌کند نتایج پژوهشی را به‌صورت **قابل‌تکرار (Reproducible)** تولید کند.

NeuroBottleneck aims to produce research results in a **reproducible** manner.

هر آزمایش نهایی باید تا حد امکان شامل موارد زیر باشد:

Each final experiment should, whenever possible, include:

```text
Dataset / Scenario
       +
Network Topology
       +
Configuration
       +
Random Seed
       +
Model Version
       +
Training Parameters
       +
Evaluation Protocol
       ↓
Reproducible Result
```

بنابراین، نتایج بدون مشخص بودن configuration، سناریوی آزمایش، روش ارزیابی و artifactهای لازم به‌عنوان نتیجه‌ی نهایی پژوهش در نظر گرفته نخواهند شد.

Therefore, results without a documented configuration, experimental scenario, evaluation protocol, and necessary artifacts will not be considered final research results.

---

# 🧩 اصول مهندسی پروژه | Engineering Principles

### Modular Design

**اجزای پروژه باید تا حد امکان مستقل، قابل‌آزمایش و قابل‌جایگزینی باشند.**

**Project components should remain as independent, testable, and replaceable as possible.**

### Reproducibility

**آزمایش‌ها باید با configuration مشخص و قابل بازتولید اجرا شوند.**

**Experiments should be executed using explicit and reproducible configurations.**

### Research Integrity

**هیچ metric یا performance claim بدون اجرای واقعی آزمایش و ثبت روش ارزیابی در README یا گزارش پژوهشی ارائه نخواهد شد.**

**No metric or performance claim will be reported without an actual experiment and a documented evaluation methodology.**

### Separation of Concerns

**کد محیط، مدل، آموزش، ارزیابی و تحلیل نتایج تا حد امکان از یکدیگر تفکیک می‌شوند.**

**Environment, modeling, training, evaluation, and analysis components should remain separated wherever practical.**

### Experimental Baselines

**هر روش جدید باید در برابر baselineهای مشخص و قابل‌تکرار ارزیابی شود.**

**New methods should be evaluated against explicit and reproducible baselines.**

---

# 📡 ارتباط با O-RAN | O-RAN Direction

NeuroBottleneck از ابتدا با نگاه به معماری‌های شبکه‌ی مدرن و به‌خصوص **O-RAN-oriented intelligent networking** طراحی شده است.

NeuroBottleneck is designed with modern network architectures and particularly **O-RAN-oriented intelligent networking** as a long-term direction.

با این حال، در وضعیت فعلی پروژه نباید ادعا کرد که یک **O-RAN production controller** یا سامانه‌ی عملیاتی متصل به شبکه‌ی واقعی است.

However, the current project should not be considered an **O-RAN production controller** or an operational system connected to a live carrier network.

هدف فعلی ایجاد هسته‌ی الگوریتمی و محیط آزمایشگاهی است که در مراحل بعد بتواند به مفاهیمی مانند **RIC، xApp و rApp** نگاشت داده شود.

The current objective is to establish the algorithmic core and experimental environment that can later be mapped to concepts such as **RIC, xApp, and rApp**.

---

# 🔭 هدف نهایی | Long-Term Goal

هدف نهایی NeuroBottleneck ساخت یک نمونه‌ی پژوهشی از یک **Network Intelligence Platform** است که بتواند وضعیت شبکه را از طریق یک نمایش گرافی دریافت کند، نقاط بحرانی را شناسایی کند، وضعیت شبکه را برای یک عامل هوشمند نمایش دهد و در نهایت اقدامات بهینه‌سازی را به‌صورت تطبیقی پیشنهاد یا انتخاب کند.

The long-term goal of NeuroBottleneck is to develop a research prototype of a **Network Intelligence Platform** capable of receiving network state through a graph representation, identifying critical components, providing an appropriate state representation to an intelligent agent, and eventually recommending or selecting adaptive optimization actions.

چشم‌انداز نهایی پروژه، حرکت از **تحلیل ایستا (Static Analysis)** به سمت **تصمیم‌گیری تطبیقی (Adaptive Decision-Making)** و در نهایت مطالعه‌ی امکان دستیابی به سطوح بالاتر **Network Autonomy** است.

The long-term research direction is to move from **static analysis** toward **adaptive decision-making**, and eventually investigate higher levels of **network autonomy**.

```text
                OBSERVE
                   │
                   ▼
          ┌─────────────────┐
          │ Network State   │
          └────────┬────────┘
                   │
                   ▼
                MODEL
                   │
                   ▼
          ┌─────────────────┐
          │ Graph / GNN     │
          └────────┬────────┘
                   │
                   ▼
               ANALYZE
                   │
                   ▼
          ┌─────────────────┐
          │ Bottleneck /    │
          │ Criticality     │
          └────────┬────────┘
                   │
                   ▼
               DECIDE
                   │
                   ▼
          ┌─────────────────┐
          │ DRL Agent       │
          └────────┬────────┘
                   │
                   ▼
                ACT
                   │
                   ▼
          ┌─────────────────┐
          │ Optimization    │
          │ Action          │
          └────────┬────────┘
                   │
                   ▼
               EVALUATE
                   │
                   └──────────────► OBSERVE
```

---

# 📈 تعریف موفقیت پروژه | Definition of Success

موفقیت NeuroBottleneck صرفاً با «داشتن یک مدل یادگیری عمیق» تعریف نمی‌شود.

Success in NeuroBottleneck is not defined merely by having a deep-learning model.

یک نتیجه‌ی پژوهشی موفق باید بتواند حداقل چهار ویژگی را نشان دهد:

A successful research result should demonstrate at least four properties:

1. **قابل‌اندازه‌گیری بودن | Measurability**  
   عملکرد سیستم با KPIهای مشخص ارزیابی شود.  
   Performance is evaluated using explicit KPIs.

2. **مقایسه‌پذیری | Comparability**  
   روش پیشنهادی با baselineهای مشخص مقایسه شود.  
   The proposed method is compared against defined baselines.

3. **تکرارپذیری | Reproducibility**  
   آزمایش‌ها با configuration و seed مشخص قابل تکرار باشند.  
   Experiments can be reproduced using documented configurations and seeds.

4. **تعمیم‌پذیری | Generalization**  
   عملکرد تنها به یک توپولوژی یا یک سناریوی آموزشی محدود نباشد.  
   Performance is not restricted to a single topology or training scenario.

---

# 🗂️ ساختار مفهومی پروژه | Conceptual Project Structure

ساختار repository به‌صورت تدریجی و متناسب با رشد پروژه تکمیل خواهد شد.

The repository structure will evolve incrementally as the project matures.

```text
NeuroBottleneck/
│
├── code/
│   ├── environment/
│   ├── graph/
│   ├── models/
│   ├── rl/
│   ├── evaluation/
│   └── utils/
│
├── configs/
├── datasets/
├── experiments/
├── papers/
├── notes/
├── diagrams/
├── reports/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🧭 مسیر توسعه | Development Direction

**مسیر توسعه پروژه به‌صورت کلی از یک pipeline تحلیلی به سمت یک حلقه‌ی تصمیم‌گیری هوشمند حرکت می‌کند.**

**The overall development path moves from an analytical pipeline toward an intelligent closed-loop decision system.**

```text
Phase 1
Research Foundation
       ↓
Phase 2
Environment + Baselines
       ↓
Phase 3
GNN + DRL Optimization
       ↓
Phase 4
Robustness + Generalization
       ↓
Research Prototype
       ↓
O-RAN-Oriented Network Intelligence
```

---

# 📚 حوزه‌های پژوهشی | Research Themes

- Network Optimization
- Network Resilience
- Bottleneck Detection
- Critical Node / Link Analysis
- Graph Neural Networks
- Graph Representation Learning
- Deep Reinforcement Learning
- Dynamic Traffic Management
- Resource Optimization
- Network Autonomy
- O-RAN-Oriented Intelligence
- Reproducible Network Research

---

# 👤 نویسنده | Author

**Mohammad Mahdi Shafighi**  
**M.Sc. Artificial Intelligence**

NeuroBottleneck is developed as a research-oriented project for studying intelligent network optimization, graph-based network intelligence, and adaptive decision-making.

---

## 📌 وضعیت فعلی | Current Status

> **Phase 2 — Environment & Baseline Modeling**
>
> **Mid-Phase / Active Development**

**پروژه در حال توسعه است و نتایج عددی نهایی تا زمان تکمیل آزمایش‌های کنترل‌شده و قابل‌تکرار به‌عنوان نتیجه‌ی قطعی پژوهش در نظر گرفته نمی‌شوند.**

**The project is under active development, and final numerical results will not be considered conclusive research findings until controlled and reproducible experiments have been completed.**

---

## 🚀 Vision

> **Observe the Network.  
> Model the Network.  
> Understand the Bottleneck.  
> Decide Intelligently.  
> Act Adaptively.  
> Measure the Result.**
>
> **مشاهده‌ی شبکه.  
> مدل‌سازی شبکه.  
> درک گلوگاه.  
> تصمیم‌گیری هوشمند.  
> اقدام تطبیقی.  
> اندازه‌گیری نتیجه.**
