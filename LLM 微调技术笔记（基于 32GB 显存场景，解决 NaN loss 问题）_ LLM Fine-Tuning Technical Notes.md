# LLM 微调技术笔记（基于 32GB 显存场景，解决 NaN loss 问题）| LLM Fine-Tuning Technical Notes (Based on 32GB VRAM Scenario, Solving NaN Loss Issue)

# 一、背景与核心问题 | I. Background and Core Issues

场景：使用 32GB 显存 GPU，尝试微调 Qwen3-8B 模型，遇到两个核心问题：① 显存不足，无法完成微调；② 训练过程中出现 NaN loss，训练中断。| Scenario: When attempting to fine-tune the Qwen3-8B model using a GPU with 32GB VRAM, two core issues are encountered: ① Insufficient VRAM to complete fine-tuning; ② NaN loss occurs during training, causing training interruption.

核心需求：在 32GB 显存限制下，成功微调 Qwen3-8B 模型，解决 NaN loss 问题，确保训练稳定。| Core Requirement: Successfully fine-tune the Qwen3-8B model under the 32GB VRAM limit, solve the NaN loss issue, and ensure stable training.

# 二、关键技术术语解释（零基础必看，含来龙去脉、原理逻辑+英文名词）| II. Explanation of Key Technical Terms (Must-read for Beginners, Including Origin, Principle Logic + English Terms)

- **bit（比特）**：| **bit (Binary Digit)**：
**英文全称**：bit（Binary Digit，二进制数字）| **English Full Name**: bit (Binary Digit)
        
**来龙去脉**：bit 是计算机存储和处理数据的最小单位，最早由数学家克劳德·香农提出，用于表示二进制中的“0”和“1”。随着计算机技术发展，bit 成为衡量数据精度、显存占用的核心单位——在 LLM 中，模型权重的精度（如 32bit、16bit、4bit）本质就是用多少个 bit 来表示一个数值，直接决定了显存占用和数值稳定性，是量化、精度设置的基础概念。| **Origin**: The bit is the smallest unit for computers to store and process data, first proposed by mathematician Claude Shannon to represent "0" and "1" in binary. With the development of computer technology, the bit has become a core unit for measuring data precision and VRAM usage. In LLMs, the precision of model weights (e.g., 32bit, 16bit, 4bit) essentially refers to how many bits are used to represent a value, which directly determines VRAM usage and numerical stability, and is a basic concept for quantization and precision setting.
**核心原理**：bit 的核心是“二进制编码”，1 个 bit 只能表示 2 种状态（0 或 1）；n 个 bit 可以表示 2ⁿ 种状态。在 LLM 中，权重数值（如 0.123）需要通过二进制编码存储，bit 数越多，能表示的数值范围越广、精度越高，但占用的显存也越多（1 个 32bit 数值占用 4 字节显存，1 个 16bit 数值占用 2 字节，1 个 4bit 数值仅占用 0.5 字节）。| **Core Principle**: The core of a bit is "binary encoding". One bit can only represent 2 states (0 or 1); n bits can represent 2ⁿ states. In LLMs, weight values (e.g., 0.123) need to be stored through binary encoding. The more bits, the wider the range of values that can be represented and the higher the precision, but the more VRAM is occupied (one 32bit value occupies 4 bytes of VRAM, one 16bit value occupies 2 bytes, and one 4bit value occupies only 0.5 bytes).
        
**通俗解释**：bit 就是计算机“存储数据的最小积木”，积木（bit）越多，能精准表示的数值越全，但占用的“空间”（显存）也越大；反之，积木越少，空间越省，但精准度会略有牺牲。| **Popular Explanation**: A bit is the "smallest building block for computers to store data". The more building blocks (bits), the more complete the values that can be accurately represented, but the more "space" (VRAM) is occupied; on the contrary, the fewer building blocks, the more space is saved, but the precision will be slightly sacrificed.
      

- **FFT（Full Fine-Tuning，全量微调）**：| **FFT (Full Fine-Tuning)**：
        
**英文全称**：Full Fine-Tuning | **English Full Name**: Full Fine-Tuning
        
**来龙去脉**：早期 LLM 微调技术较为简单，为了让模型完全适配新任务，会更新模型所有权重（比如 Qwen3-8B 有 80 亿个权重参数）。这种方式的优势是拟合效果好，但缺点是显存占用极高——每一个权重参数都需要存储梯度、优化器状态，32GB 显存根本无法支撑 8B 级模型的全量微调，因此逐渐被更高效的微调方式替代。| **Origin**: Early LLM fine-tuning technology was relatively simple. To fully adapt the model to new tasks, all weights of the model would be updated (for example, Qwen3-8B has 8 billion weight parameters). The advantage of this method is good fitting effect, but the disadvantage is extremely high VRAM usage—each weight parameter needs to store gradients and optimizer states, and 32GB VRAM cannot support full fine-tuning of 8B-level models at all, so it is gradually replaced by more efficient fine-tuning methods.
       
**核心原理**：微调时，模型的所有参数（权重矩阵、偏置项等）都会根据新任务的训练数据进行更新，训练过程中需要同时存储“原始权重”“梯度值”“优化器状态（如 AdamW 的动量项）”，三者的显存占用叠加，导致 8B 级模型在 32GB 显存下无法运行。简单说，就是“从头到尾修改模型，需要存储的中间数据太多”。| **Core Principle**: During fine-tuning, all parameters of the model (weight matrices, bias terms, etc.) will be updated according to the training data of the new task. During training, it is necessary to store "original weights", "gradient values", and "optimizer states (such as momentum terms of AdamW)" at the same time. The superposition of their VRAM usage makes it impossible for 8B-level models to run on 32GB VRAM. Simply put, it is "modifying the model from start to finish, requiring too much intermediate data to be stored".
        
**通俗解释**：微调时“从头到尾改模型”，效果好但太费显存，32GB 显存跑不动 Qwen3-8B。| **Popular Explanation**: "Modify the model from start to finish" during fine-tuning, which has a good effect but consumes too much VRAM—32GB VRAM cannot run Qwen3-8B.
      

- **LoRA（Low-Rank Adaptation，低秩适配）**：| **LoRA (Low-Rank Adaptation)**：
**英文全称**：Low-Rank Adaptation | **English Full Name**: Low-Rank Adaptation
        
**来龙去脉**：随着模型规模增大（从 1B 到 70B+），全量微调的显存压力越来越大，研究人员发现“模型权重的变化具有低秩特性”——不需要改所有权重，只要在原有权重上新增两个小型低秩矩阵（A 和 B），优化这两个小矩阵（仅占模型权重的 1%），就能达到接近全量微调的效果。LoRA 由此诞生，核心是“用少量参数更新，替代全量权重更新”，大幅节省显存。| **Origin**: As the model scale increases (from 1B to 70B+), the VRAM pressure of full fine-tuning becomes greater and greater. Researchers found that "the changes in model weights have low-rank characteristics"—instead of modifying all weights, it is only necessary to add two small low-rank matrices (A and B) to the original weights and optimize these two small matrices (accounting for only 1% of the model weights) to achieve an effect close to full fine-tuning. LoRA was thus born, with the core of "using a small number of parameter updates to replace full weight updates" to greatly save VRAM.
        
**核心原理**：利用“低秩分解”原理——任何一个复杂的高维权重矩阵（比如 768×768），都可以分解为两个低维矩阵（比如 768×16 和 16×768）的乘积，这两个低维矩阵就是 LoRA 新增的参数。微调时，只更新这两个小矩阵，不改变原始权重，因此显存占用仅为全量微调的 1%~5%，同时能保证微调效果接近全量微调。| **Core Principle**: Using the principle of "low-rank decomposition"—any complex high-dimensional weight matrix (such as 768×768) can be decomposed into the product of two low-dimensional matrices (such as 768×16 and 16×768), and these two low-dimensional matrices are the new parameters of LoRA. During fine-tuning, only these two small matrices are updated without changing the original weights, so the VRAM usage is only 1%~5% of full fine-tuning, and the fine-tuning effect can be guaranteed to be close to full fine-tuning.
        
**通俗解释**：微调时“只改模型的一小部分（两个小矩阵）”，省显存还能达到接近全量微调的效果。| **Popular Explanation**: "Only modify a small part of the model (two small matrices)" during fine-tuning, which saves VRAM and can achieve an effect close to full fine-tuning.
      

- **QLoRA**：| **QLoRA**：
        
**英文全称**：Quantized LoRA（量化低秩适配）| **English Full Name**: Quantized LoRA
        
**来龙去脉**：LoRA 虽然省显存，但对于 70B 甚至更大的模型，即使只优化 1% 的参数，16bit 精度下的显存占用依然很高。研究人员便将“LoRA 高效微调”与“4bit/8bit 量化”结合，形成 QLoRA——先将模型权重从 16bit 压缩到 4bit/8bit（节省 75% 显存），再用 LoRA 微调新增的低秩矩阵。它解决了“大模型微调显存不足”的核心痛点，是 32GB 显存微调 8B 级模型的最优方案（Unsloth 框架对 QLoRA 做了优化，大幅减少精度损失）。| **Origin**: Although LoRA saves VRAM, for 70B or even larger models, even if only 1% of the parameters are optimized, the VRAM usage under 16bit precision is still very high. Researchers combined "LoRA efficient fine-tuning" with "4bit/8bit quantization" to form QLoRA—first compress the model weights from 16bit to 4bit/8bit (saving 75% of VRAM), then use LoRA to fine-tune the newly added low-rank matrices. It solves the core pain point of "insufficient VRAM for large model fine-tuning" and is the optimal solution for fine-tuning 8B-level models with 32GB VRAM (the Unsloth framework optimizes QLoRA to greatly reduce precision loss).
        
**核心原理**：QLoRA = 4bit/8bit 量化 + LoRA 微调。第一步，用量化工具（如 bitsandbytes）将模型原始 16bit 权重压缩为 4bit/8bit（通过“权重映射”将高精度数值压缩到低精度范围，牺牲少量精度）；第二步，在量化后的模型上，新增 LoRA 低秩矩阵进行微调，既保留了 LoRA 省参数的优势，又通过量化进一步降低显存占用，实现“低显存+高精度”的平衡。|**Core Principle**: QLoRA = 4bit/8bit Quantization + LoRA Fine-Tuning. Step 1: Use a quantization tool (such as bitsandbytes) to compress the original 16bit model weights to 4bit/8bit (compress high-precision values to the low-precision range through "weight mapping", sacrificing a small amount of precision); Step 2: Add LoRA low-rank matrices to the quantized model for fine-tuning, which not only retains the advantage of LoRA's parameter saving but also further reduces VRAM usage through quantization, achieving a balance between "low VRAM and high precision".
        
**通俗解释**：在 LoRA 基础上，先把模型“压缩”（4bit/8bit），再微调，显存占用再降一级，32GB 显存刚好能支撑。| **Popular Explanation**: On the basis of LoRA, first "compress" the model (4bit/8bit), then fine-tune it, which further reduces VRAM usage—32GB VRAM can just support it.

- **量化（Quantization）**：| **Quantization**：
        
**英文全称**：Quantization | **English Full Name**: Quantization
        
**来龙去脉**：模型权重默认是 32bit（float32），能精准表示数值，但占用显存多；16bit（bfloat16/float16）显存占用减半，但精度略有损失；4bit/8bit 则是进一步压缩——核心逻辑是“用牺牲少量精度，换取大幅显存节省”。量化技术的出现，就是为了解决“大模型显存占用过高，无法在普通 GPU 上运行/微调”的问题，常用工具是 bitsandbytes（专门用于 LLM 量化，兼容性好）。| **Origin**: The default model weight precision is 32bit (float32), which can accurately represent values but occupies a lot of VRAM; 16bit (bfloat16/float16) halves the VRAM usage but has a slight loss of precision; 4bit/8bit is further compression—the core logic is "sacrificing a small amount of precision in exchange for a significant reduction in VRAM usage". The emergence of quantization technology is to solve the problem of "excessively high VRAM usage of large models, making them unable to run/fine-tune on ordinary GPUs". A common tool is bitsandbytes (specialized for LLM quantization with good compatibility).
        
**核心原理**：核心是“数值映射与精度取舍”。模型权重的数值范围通常在 [-1, 1] 之间，量化时会将这个范围内的高精度数值（如 32bit 的 0.123456789），映射到低精度的离散数值（如 4bit 的 0.125），通过“舍入误差”牺牲少量精度，换取显存占用的大幅降低（比如 32bit → 4bit，显存占用变为原来的 1/8）。LLM 常用的量化方式是“整数量化”，将浮点型权重转换为整数型，进一步节省显存。| **Core Principle**: The core is "value mapping and precision trade-off". The value range of model weights is usually between [-1, 1]. During quantization, high-precision values in this range (such as 0.123456789 in 32bit) are mapped to low-precision discrete values (such as 0.125 in 4bit), and a small amount of precision is sacrificed through "rounding error" in exchange for a significant reduction in VRAM usage (for example, 32bit → 4bit, VRAM usage becomes 1/8 of the original). The common quantization method for LLMs is "integer quantization", which converts floating-point weights to integer type to further save VRAM.
        
**通俗解释**：把模型的“数值精度降低”（比如从 32bit 降到 4bit），像压缩文件一样省显存，代价是精度略有下降（Unsloth 优化后，4bit 精度损失已基本可忽略）。| **Popular Explanation**: "Reduce the numerical precision of the model" (for example, from 32bit to 4bit), which saves VRAM like compressing files, at the cost of a slight loss of precision (after Unsloth optimization, the 4bit precision loss is basically negligible).
      

- **Unsloth**：| **Unsloth**：
        
**英文全称**：Unsloth（无官方中文译名，直译“无延迟/快速”）| **English Full Name**: Unsloth (No official Chinese translation, literal translation: "No Delay/Fast")
        
**来龙去脉**：随着 LLM 普及，很多研究者、开发者想微调模型，但缺乏高显存 GPU（比如只有 3GB、8GB 显存），且传统微调框架（如 Hugging Face Transformers）配置复杂、速度慢。Unsloth 框架应运而生，它是一款轻量级 LLM 微调工具，核心优势是“优化显存占用和训练速度”，原生支持 LoRA、QLoRA 等高效微调方式，安装简单（pip 一键安装），甚至 3GB 显存就能微调 8B 模型，解决了“普通用户微调大模型难”的问题。| **Origin**: With the popularization of LLMs, many researchers and developers want to fine-tune models but lack high-VRAM GPUs (such as only 3GB or 8GB VRAM), and traditional fine-tuning frameworks (such as Hugging Face Transformers) are complex to configure and slow. The Unsloth framework emerged as the times require. It is a lightweight LLM fine-tuning tool with core advantages of "optimizing VRAM usage and training speed". It natively supports efficient fine-tuning methods such as LoRA and QLoRA, is simple to install (one-click pip installation), and can even fine-tune 8B models with 3GB VRAM, solving the problem of "difficulty for ordinary users to fine-tune large models".
        
**核心原理**：Unsloth 基于 Hugging Face Transformers 框架优化，核心做了三点改进：① 优化模型加载逻辑，支持“量化+LoRA”一键配置，减少冗余显存占用；② 优化训练循环，减少中间变量的存储，提升训练速度；③ 内置高精度 4bit 量化模型（如 unsloth-bnb-4bit），通过“量化校准”减少精度损失，让 4bit 量化模型的性能接近 16bit 模型。| **Core Principle**: Unsloth is optimized based on the Hugging Face Transformers framework, with three core improvements: ① Optimize model loading logic to support one-click configuration of "quantization + LoRA" and reduce redundant VRAM usage; ② Optimize the training loop to reduce the storage of intermediate variables and improve training speed; ③ Built-in high-precision 4bit quantized models (such as unsloth-bnb-4bit), which reduce precision loss through "quantization calibration" and make the performance of 4bit quantized models close to 16bit models.
        
**通俗解释**：专门为“显存小、零基础”用户设计的微调框架，简化操作、省显存、提速度，不用复杂配置就能上手。| **Popular Explanation**: A fine-tuning framework specially designed for users with "small VRAM and zero foundation", which simplifies operations, saves VRAM, improves speed, and can be used without complex configuration.
      

- **AMP Autocast（自动混合精度训练）**：| **AMP Autocast (Automatic Mixed Precision Training)**：
        
**英文全称**：Automatic Mixed Precision Autocast | **English Full Name**: Automatic Mixed Precision Autocast
        
**来龙去脉**：为了平衡“显存占用”和“训练精度”，研究人员提出混合精度训练——训练时，模型计算用 16bit（省显存、提速度），梯度计算用 32bit（保精度，避免梯度消失/爆炸）。AMP Autocast 是 PyTorch 内置的功能，能自动切换精度，不用手动配置。但它有一个关键前提：全程精度要统一，若手动强制转换精度（比如把 16bit 转 32bit），会破坏这种平衡，导致数值不稳定。| **Origin**: To balance "VRAM usage" and "training precision", researchers proposed mixed-precision training—during training, model calculations use 16bit (saving VRAM and improving speed), and gradient calculations use 32bit (ensuring precision and avoiding gradient vanishing/explosion). AMP Autocast is a built-in function of PyTorch that can automatically switch precision without manual configuration. However, it has a key premise: the precision must be consistent throughout. If the precision is manually forced to be converted (such as converting 16bit to 32bit), this balance will be destroyed, leading to numerical instability.
**核心原理**：AMP Autocast 会自动识别模型中的“计算密集型操作”（如矩阵乘法），用 16bit 精度计算（省显存、提速度）；识别“精度敏感型操作”（如梯度计算、损失计算），用 32bit 精度计算（保精度）。它会自动管理精度转换，无需手动干预，但若手动强制转换精度（如 .float() 转 32bit），会打破这种自动平衡，导致数值计算异常。| **Core Principle**: AMP Autocast will automatically identify "computationally intensive operations" in the model (such as matrix multiplication) and use 16bit precision for calculation (saving VRAM and improving speed); it will identify "precision-sensitive operations" (such as gradient calculation and loss calculation) and use 32bit precision for calculation (ensuring precision). It automatically manages precision conversion without manual intervention, but if the precision is manually forced to be converted (such as .float() to 32bit), this automatic balance will be broken, leading to abnormal numerical calculation.
        
**通俗解释**：自动切换“省显存的精度（16bit）”和“保精度的精度（32bit）”，但不能手动干预精度转换，否则会出问题。| **Popular Explanation**: Automatically switch between "VRAM-saving precision (16bit)" and "precision-guaranteeing precision (32bit)", but manual intervention in precision conversion is not allowed, otherwise problems will occur.
      

- **bfloat16/float32**：| **bfloat16/float32**：
       
**英文全称**：bfloat16（Brain Floating Point 16-bit）、float32（Single-Precision Floating Point）| **English Full Name**: bfloat16 (Brain Floating Point 16-bit), float32 (Single-Precision Floating Point)
        
**来龙去脉**：两者都是数值精度的表示方式，核心区别是“显存占用”和“数值范围”。float32（32bit）是早期默认精度，数值范围广、精度高，但显存占用大；随着大模型出现，float16（16bit）应运而生，显存占用减半，但数值范围小，容易出现下溢出；bfloat16（16bit）是改进版，保留了 float32 的数值范围，同时兼顾 16bit 的显存优势，数值稳定性更好，成为目前大模型训练的首选精度。| **Origin**: Both are methods of representing numerical precision, with the core difference being "VRAM usage" and "value range". float32 (32bit) is the early default precision, with a wide value range and high precision but large VRAM usage; with the emergence of large models, float16 (16bit) came into being, which halves the VRAM usage but has a small value range and is prone to underflow; bfloat16 (16bit) is an improved version that retains the value range of float32 and at the same time takes into account the VRAM advantage of 16bit, with better numerical stability, and has become the preferred precision for current large model training.
        
**核心原理**：两者的本质区别是“二进制位数的分配”——float32 用 32 位二进制表示一个数值，其中 1 位表示符号、8 位表示指数、23 位表示小数，精度高、数值范围广，但占用 4 字节显存；bfloat16 用 16 位二进制表示，其中 1 位表示符号、8 位表示指数、7 位表示小数，保留了 float32 的指数范围（避免下溢出），但小数位数减少（牺牲少量精度），仅占用 2 字节显存，完美平衡显存和稳定性。| **Core Principle**: The essential difference between the two is the "allocation of binary bits"—float32 uses 32 binary bits to represent a value, of which 1 bit represents the sign, 8 bits represent the exponent, and 23 bits represent the decimal, with high precision and a wide value range but occupying 4 bytes of VRAM; bfloat16 uses 16 binary bits to represent a value, of which 1 bit represents the sign, 8 bits represent the exponent, and 7 bits represent the decimal, retaining the exponent range of float32 (avoiding underflow) but reducing the number of decimal bits (sacrificing a small amount of precision), occupying only 2 bytes of VRAM, perfectly balancing VRAM and stability.
        
**通俗解释**：bfloat16 是“兼顾省显存和稳定性”的优选精度；float32 太费显存，不适合 8B 模型训练。| **Popular Explanation**: bfloat16 is the preferred precision that "balances VRAM saving and stability"; float32 consumes too much VRAM and is not suitable for 8B model training.
      

- **NaN loss**：| **NaN loss (Not a Number loss)**：
        
**英文全称**：Not a Number loss（非数字损失）| **English Full Name**: Not a Number loss
        
**来龙去脉**：NaN（非数字）是训练中常见的错误，核心成因是“数值计算异常”。在 LLM 微调中，最常见的原因是“精度不匹配导致的下溢出”——当数值过小，超出当前精度能表示的范围时，会被识别为“非数字”，导致 loss 变为 NaN，训练中断。比如你之前的操作：AMP 用 16bit 训练，却手动把张量转成 32bit，两种精度冲突，数值超出 16bit 范围，就会出现 NaN loss。| **Origin**: NaN (Not a Number) is a common error in training, with the core cause being "abnormal numerical calculation". In LLM fine-tuning, the most common reason is "underflow caused by precision mismatch"—when the value is too small, exceeding the range that can be represented by the current precision, it will be identified as "Not a Number", causing the loss to become NaN and training to interrupt. For example, your previous operation: AMP uses 16bit for training, but you manually convert the tensor to 32bit. The two precisions conflict, and the value exceeds the 16bit range, resulting in NaN loss.
        
**核心原理**：NaN 的本质是“数值计算超出当前精度的表示范围”，常见于两种情况：① 下溢出（underflow）：数值过小，小于当前精度能表示的最小值（如 16bit 精度无法表示 1e-45 以下的数值），会被识别为 NaN；② 上溢出（overflow）：数值过大，超出精度范围。你遇到的情况是前者——AMP 用 16bit 计算，手动转 32bit 后，数值被拉伸，部分小数超出 16bit 范围，导致下溢出，loss 变为 NaN。| **Core Principle**: The essence of NaN is "numerical calculation exceeding the representable range of the current precision", which is common in two cases: ① Underflow: the value is too small, smaller than the minimum value that can be represented by the current precision (for example, 16bit precision cannot represent values below 1e-45), which will be identified as NaN; ② Overflow: the value is too large, exceeding the precision range. Your case is the former—after AMP calculates with 16bit and you manually convert it to 32bit, the value is stretched, and some decimals exceed the 16bit range, leading to underflow and the loss becoming NaN.
        
**通俗解释**：训练时计算出“无效数字”，本质是精度不匹配导致的，会直接让训练停掉。| **Popular Explanation**: "Invalid numbers" are calculated during training, which is essentially caused by precision mismatch and will directly stop training.

- **下溢出（Underflow）**：| **Underflow**：
        
**英文全称**：Underflow | **English Full Name**: Underflow
        
**来龙去脉**：下溢出是数值计算中的常见异常，最早出现在传统计算机编程中，随着大模型的普及，因精度设置不当（如混合精度冲突、低精度量化），下溢出在 LLM 微调中变得尤为常见，也是导致 NaN loss 的核心原因之一。它与“上溢出（Overflow，数值过大超出范围）”相对，专门指“数值过小，超出当前精度能表示的最小值”的异常情况。| **Origin**: Underflow is a common anomaly in numerical calculation, which first appeared in traditional computer programming. With the popularization of large models, due to improper precision settings (such as mixed precision conflict and low-precision quantization), underflow has become particularly common in LLM fine-tuning and is one of the core causes of NaN loss. It is relative to "overflow (the value is too large to exceed the range)", specifically referring to the abnormal situation where "the value is too small to exceed the minimum value that can be represented by the current precision".
        
**核心原理**：每一种精度（如 32bit、16bit、4bit）都有固定的“数值表示范围”，当计算出的数值小于这个范围的最小值时，就会发生下溢出——此时计算机无法用当前精度表示该数值，会将其识别为“非数字（NaN）”或“零”，导致后续计算异常（如 LLM 训练中 loss 变为 NaN，训练中断）。比如 16bit 精度的最小正数约为 1e-45，若计算出 1e-50 的数值，就会发生下溢出。| **Core Principle**: Each precision (such as 32bit, 16bit, 4bit) has a fixed "value representation range". When the calculated value is smaller than the minimum value of this range, underflow occurs—at this time, the computer cannot represent the value with the current precision and will identify it as "Not a Number (NaN)" or "zero", leading to subsequent calculation anomalies (such as the loss becoming NaN during LLM training and training interruption). For example, the minimum positive number of 16bit precision is about 1e-45, and if a value of 1e-50 is calculated, underflow will occur.
       
**通俗解释**：下溢出就是“数值太小了，超出了当前精度能‘记住’的最小范围”，计算机无法识别这个极小的数，就会把它标记为“无效数字”，进而导致训练出错。| **Popular Explanation**: Underflow means "the value is too small, exceeding the minimum range that the current precision can 'remember'". The computer cannot recognize this extremely small number and will mark it as an "invalid number", which in turn causes training errors.
      

- **Gradient Checkpointing（梯度检查点）**：| **Gradient Checkpointing**：
        
**英文全称**：Gradient Checkpointing | **English Full Name**: Gradient Checkpointing
       
**来龙去脉**：即使使用 QLoRA 和量化，微调 8B 模型时，梯度存储依然会占用一定显存。梯度检查点技术的核心逻辑是“牺牲少量计算速度，换取显存节省”——训练时不存储所有中间梯度，只存储关键“检查点”的梯度，需要时再通过计算恢复其他梯度，从而大幅降低梯度存储的显存占用，是微调大模型时的“额外显存优化技巧”。| **Origin**: Even with QLoRA and quantization, gradient storage still occupies a certain amount of VRAM when fine-tuning 8B models. The core logic of gradient checkpointing technology is "sacrificing a small amount of calculation speed in exchange for VRAM saving"—during training, not all intermediate gradients are stored, only the gradients of key "checkpoints" are stored, and other gradients are recovered through calculation when needed, thereby greatly reducing the VRAM occupancy of gradient storage. It is an "additional VRAM optimization technique" when fine-tuning large models.
        
**核心原理**：正常训练时，模型会存储每一层的中间输出（用于计算梯度），这会占用大量显存；梯度检查点会“舍弃部分中间输出”，只保留几个关键的“检查点”输出，当需要计算梯度时，通过反向传播重新计算被舍弃的中间输出，从而减少中间输出的存储量，节省显存。代价是需要额外计算被舍弃的中间输出，训练速度会降低 10%~20%。| **Core Principle**: During normal training, the model stores the intermediate output of each layer (for gradient calculation), which occupies a lot of VRAM; gradient checkpointing will "discard some intermediate outputs" and only retain the outputs of a few key "checkpoints". When gradients need to be calculated, the discarded intermediate outputs are recalculated through backpropagation, thereby reducing the storage of intermediate outputs and saving VRAM. The cost is the need to additionally calculate the discarded intermediate outputs, and the training speed will be reduced by 10%~20%.
        
**通俗解释**：训练时“不存所有中间数据，只存关键数据”，省显存但训练速度会稍慢一点。| **Popular Explanation**: "Do not store all intermediate data, only key data" during training, which saves VRAM but slows down the training speed a little.
     

# 三、显存不足问题的解决方案（Paolo 推荐方案）| III. Solutions to Insufficient VRAM (Paolo's Recommended Plan)

## 3.1 核心结论 | 3.1 Core Conclusion

32GB 显存**无法支撑 Qwen3-8B 全量微调（FFT）**，必须采用“QLoRA 量化微调 + Unsloth 框架”，结合 4bit/8bit 量化，才能将显存占用控制在可承受范围。| **32GB VRAM cannot support full fine-tuning (FFT) of Qwen3-8B**. It is necessary to adopt "QLoRA quantization fine-tuning + Unsloth framework" combined with 4bit/8bit quantization to control the VRAM usage within an acceptable range.

## 3.2 具体实施步骤 | 3.2 Specific Implementation Steps

1. **安装 Unsloth 框架**：直接通过 pip 安装，命令：`pip install unsloth`（支持 Linux、WSL、Windows 系统）。| **Install the Unsloth framework**: Install directly via pip with the command: `pip install unsloth` (supports Linux, WSL, and Windows systems).

2. **采用 QLoRA 微调**：结合 bitsandbytes 工具，将模型量化为 4bit/8bit（优先 4bit，显存节省更多），同时使用 LoRA 高效微调，仅训练少量低秩矩阵。| **Adopt QLoRA fine-tuning**: Combine the bitsandbytes tool to quantize the model to 4bit/8bit (4bit is preferred for more VRAM saving), and use LoRA for efficient fine-tuning, only training a small number of low-rank matrices.

3. **模型加载建议**：使用 Unsloth 提供的 4bit 量化模型（如模型名含“unsloth-bnb-4bit”），这类模型比普通 4bit 模型精度更高，显存占用更低。|**Model Loading Suggestion**: Use the 4bit quantized model provided by Unsloth (such as models with "unsloth-bnb-4bit" in their names), which have higher precision and lower VRAM usage than ordinary 4bit models.

4. **额外显存优化**：开启 Gradient Checkpointing（梯度检查点），进一步降低显存占用，可咨询 LLM 获取具体实现代码（支持 Unsloth 或原生 PyTorch）。| **Additional VRAM Optimization**: Enable Gradient Checkpointing to further reduce VRAM usage. You can consult an LLM for specific implementation code (supports Unsloth or native PyTorch).

# 四、NaN loss 问题的根源与解决方案 | IV. Root Cause and Solutions of NaN Loss Issue

## 4.1 问题根源（精准定位）| 4.1 Root Cause (Precise Positioning)

训练中出现 NaN loss，核心原因是 **精度不匹配导致的数值下溢出（underflow）**，具体操作错误如下：| The core cause of NaN loss during training is **numerical underflow caused by precision mismatch**, with the following specific operational errors:

- 开启了 AMP Autocast（自动混合精度，默认使用 16bit 精度训练）；| Enabled AMP Autocast (automatic mixed precision, which uses 16bit precision for training by default);

- 手动使用`.float()` 将模型输出的特征张量（vit_feat、xlmr_feat）强制转换为 float32（32bit）；| Manually used `.float()` to force-convert the feature tensors (vit_feat, xlmr_feat) output by the model to float32 (32bit);

- 两种精度冲突，导致数值过小（超出 16bit 精度范围），出现下溢出，最终计算出 NaN loss。| The two precisions conflict, causing the value to be too small (exceeding the 16bit precision range), resulting in underflow and finally calculating NaN loss.

## 4.2 解决方案（Paolo 推荐，已验证可行）| 4.2 Solutions (Paolo's Recommended, Verified Feasible)

1. **移除手动精度转换**：删除代码中所有 `.float()` 强制转换操作（如 `vit_feat = vit_out.last_hidden_state[:, 0, :].float()` → 改为 `vit_feat = vit_out.last_hidden_state[:, 0, :]`）。| **Remove manual precision conversion**: Delete all `.float()` forced conversion operations in the code (e.g., `vit_feat = vit_out.last_hidden_state[:, 0, :].float()` → changed to `vit_feat = vit_out.last_hidden_state[:, 0, :]`).

2. **统一模型精度为 bfloat16**：全程使用 bfloat16 精度加载模型、训练，不使用 float32，避免精度冲突。| **Unify model precision to bfloat16**: Load and train the model with bfloat16 precision throughout, and do not use float32 to avoid precision conflicts.

3. **关闭 AMP Autocast**：由于已统一使用 bfloat16 精度，无需再开启自动混合精度，避免额外的精度干扰。|**Turn off AMP Autocast**: Since bfloat16 precision is used uniformly, there is no need to enable automatic mixed precision to avoid additional precision interference.

# 五、最终可直接复用的训练方案（完整流程）| V. Final Reusable Training Plan (Complete Process)

1. **环境准备**：安装 Unsloth 框架（`pip install unsloth`）和 bitsandbytes 量化工具。| **Environment Preparation**: Install the Unsloth framework (`pip install unsloth`) and the bitsandbytes quantization tool.

2. **模型加载**：使用 Unsloth 的 4bit 量化模型（如 Qwen3-8B 的 4bit 版本），以 bfloat16 精度加载，关闭 AMP Autocast。| **Model Loading**: Use the 4bit quantized model of Unsloth (such as the 4bit version of Qwen3-8B), load it with bfloat16 precision, and turn off AMP Autocast.

3. **微调方式**：采用 QLoRA 进行参数高效微调，不使用全量微调（FFT）。| **Fine-Tuning Method**: Adopt QLoRA for parameter-efficient fine-tuning, and do not use full fine-tuning (FFT).

4. **代码修正**：删除所有`.float()` 强制转换，确保全程精度统一为 bfloat16。| **Code Correction**: Delete all `.float()` forced conversions to ensure that the precision is uniformly bfloat16 throughout.

5. **显存优化**：开启 Gradient Checkpointing，进一步节省显存。| **VRAM Optimization**: Enable Gradient Checkpointing to further save VRAM.

6. **训练监控**：观察训练 loss，正常范围为 0.5~1.0，若 loss 为 0 需警惕过拟合，若仍出现 NaN，检查精度是否统一。| **Training Monitoring**: Observe the training loss; the normal range is 0.5~1.0. If the loss is 0, be alert to overfitting; if NaN still appears, check whether the precision is uniform.

# 六、关键注意事项 | VI. Key Notes

- 避免使用 float32 精度：float32 显存占用过高，且易与 bfloat16/AMP 产生精度冲突，导致 NaN loss。| Avoid using float32 precision: float32 has excessively high VRAM usage and is prone to precision conflicts with bfloat16/AMP, leading to NaN loss.

- 优先尝试 QLoRA，再考虑 LoRA：QLoRA 结合量化，显存占用更低，32GB 显存可稳定支撑。| Prioritize QLoRA before considering LoRA: QLoRA combined with quantization has lower VRAM usage and can be stably supported by 32GB VRAM.

- Unsloth 框架的优势：无需复杂配置，原生支持高效微调，且提供高精度 4bit 量化模型，比普通量化模型性能更好。| Advantages of the Unsloth framework: No complex configuration is required, it natively supports efficient fine-tuning, and provides high-precision 4bit quantized models with better performance than ordinary quantized models.

- 梯度检查点：若显存仍紧张，可开启该功能，牺牲少量速度换取显存节省，具体实现可咨询 LLM。| Gradient Checkpointing: If VRAM is still tight, this function can be enabled to sacrifice a small amount of speed in exchange for VRAM saving. Specific implementation can be consulted with an LLM.

# 七、参考资料 | VII. Reference Materials

Unsloth 官方微调指南：[https://unsloth.ai/docs/get-started/fine-tuning-llms-guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) | Unsloth Official Fine-Tuning Guide: [https://unsloth.ai/docs/get-started/fine-tuning-llms-guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
> （注：文档部分内容可能由 AI 生成）