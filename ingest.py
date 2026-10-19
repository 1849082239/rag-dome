# Ingest：把文档切分、向量化、灌进 Qdrant
# 对应 Dify 里"上传文档 → 分段 → 向量化"全套流程
# 文档直接嵌在脚本里，不用建文件夹、不用复制文件
import uuid
import requests

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


# ================= 配置区 =================
QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_test"  # 新建一个 collection，和旧数据隔离

EMBED_URL = "http://jacob.hidencloud.com:24576/v1/embeddings"
EMBED_KEY = "ollama"
EMBED_MODEL = "bge-m3:latest"
CHUNK_SIZE = 500  # 每段最多500字，对应 Dify 里 Maximum chunk length
# ==========================================


# 测试文档（内容为虚构，仅用于跑通流程）
DOCS = {
    "01-废钢分类标准.md": """# 废钢分类标准（测试样例，数据为虚构）

## 一、按外形尺寸与单重分类

### 1. 重型废钢
- 单件重量 ≥ 500kg，厚度 ≥ 8mm
- 主要来源：报废机械设备、钢结构拆除件、轧辊
- 要求：无封闭容器、无易燃易爆物，表面油污需清理

### 2. 中型废钢
- 单件重量 50~500kg，厚度 4~8mm
- 主要来源：汽车拆解件（车架、车桥）、农机具
- 要求：拆除油箱、电瓶等危险部件

### 3. 轻薄料
- 厚度 < 4mm，如彩钢瓦、汽车外壳、易拉罐压块
- 密度要求：压块密度 ≥ 1.2t/m³，尺寸 ≤ 600×600×800mm
- 轻薄料直接入炉出钢率低，一般需先打包或破碎处理

### 4. 破碎料
- 经破碎线处理，粒度 ≤ 120mm，密度 ≥ 1.4t/m³
- 分为一级破碎料（含铜量 < 0.25%）和二级破碎料（含铜量 0.25%~0.4%）
- 破碎料是电炉炼钢的优质原料，价格通常比重型废钢高 80~120 元/吨

## 二、按化学成分分类

| 类别 | 碳含量 | 硫含量 | 磷含量 | 备注 |
|------|--------|--------|--------|------|
| 优质碳素废钢 | ≤0.25% | ≤0.04% | ≤0.04% | 可直接用于转炉炼钢 |
| 普通废钢 | ≤0.45% | ≤0.06% | ≤0.06% | 需配比使用 |
| 合金废钢 | 含 Cr/Ni/Mo | — | — | 按合金元素单独计价 |

## 三、不合格品判定
出现以下情况整车拒收：夹带密封容器、放射性超标、含医疗废弃物、掺水掺泥超过 3%。
""",
    "02-废钢定价机制与行情.md": """# 废钢定价机制与行情（测试样例，数据为虚构）

## 一、计价方式
回收厂收购废钢采用"基础价 + 升贴水"模式：
- 基础价：以当日钢厂到货价为基准，每日 9:00 前公布
- 升水：破碎料 +100 元/吨，一级料型 +50 元/吨
- 贴水：锈蚀严重 -30 元/吨，含杂质超标按实际扣罚

## 二、杂质扣罚标准
- 含土含水：超过 2% 的部分，按 1:1 扣重
- 夹带非磁性杂物（塑料、橡胶、木块）：按检出重量的 2 倍扣罚
- 发现掺假（如钢坯中灌水泥）：整车按废杂料计价，并列入黑名单

## 三、行情影响因素
1. **钢厂检修季**：每年 6~8 月钢厂集中检修，废钢需求下降，价格通常走弱 50~100 元/吨
2. **铁矿石价格**：铁矿大涨时，电炉钢成本优势显现，废钢跟涨
3. **环保限产**：重污染天气预警期间，破碎线限产，破碎料供应收紧、价格坚挺
4. **进出口政策**：再生钢铁原料进口放开后，进口料到港量影响华东地区价格

## 四、2026 年一季度行情回顾（虚构）
- 1 月：重型废钢均价 2650 元/吨，节前备货小幅上涨
- 2 月：春节假期成交清淡，价格持稳
- 3 月：钢厂复产补库，破碎料领涨 120 元/吨
""",
    "03-环保法规与合规要求.md": """# 环保法规与合规要求（测试样例，条款为虚构简化版）

## 一、环评与排污许可
- 年处理 5 万吨以上的废钢加工企业，需编制环境影响报告表
- 破碎、分选工序需申领排污许可证，重点管理粉尘与噪声排放
- 每年 3 月底前完成上一年度排污许可执行报告

## 二、危险废物管理
废钢拆解中常见危废，须单独收集、交有资质单位处置：
- 废矿物油（油箱、变速箱残油）：危废代码 HW08
- 废油漆桶、油漆渣：危废代码 HW49
- 废铅酸电瓶：危废代码 HW31，严禁私自拆解
- 危废暂存间需防渗、防雨，贮存期限不得超过 1 年

## 三、粉尘与噪声控制
- 破碎线必须配套布袋除尘器，颗粒物排放浓度 ≤ 20mg/m³
- 厂界噪声昼间 ≤ 60dB(A)、夜间 ≤ 50dB(A)
- 堆场易扬尘物料需覆盖或喷淋，运输车辆出厂前冲洗

## 四、常见违规处罚（虚构案例警示）
- 案例 A：某厂危废露天堆放，被处 25 万元罚款并责令停产整治
- 案例 B：夜间破碎作业噪声超标，被处 8 万元罚款
- 合规建议：每季度做一次环保自查，保留除尘、危废转移台账至少 5 年
""",
    "04-质检经验与设备安全.md": """# 质检经验与设备安全（测试样例，数据为虚构）

## 一、到货质检流程
1. **登记过磅**：记录车号、货主、毛重，拍照存档
2. **人工验货**：质检员上车抽查，重点看中层和底部是否夹带杂物
3. **仪器检测**：手持光谱仪抽检合金成分，放射性检测仪全车扫描
4. **卸货复检**：卸货过程中发现掺假，立即叫停并取证
5. **结算签字**：质检单需质检员、货主双方签字确认

## 二、常见掺假手法识别
- **底部垫土**：车厢底部铺 20~30cm 泥土，上面盖好料——验货时要求看车底
- **空心钢坯灌水泥**：敲击声音发闷、重量异常——抽检时用磁铁+敲击双检
- **好坏掺半**：表层铺破碎料、里面塞轻薄料——卸货时分段验
- **洒水增重**：雨后到货重点查，含水超标按标准扣重

## 三、破碎线安全规程
- 开机前确认：除尘系统运行、皮带无人员、各急停按钮复位
- 严禁事项：运行中清理皮带滚筒、戴手套操作旋转部位、破碎腔内有人时启动
- 检修执行"断电-挂牌-上锁"三步，钥匙由检修负责人保管
- 每班检查液压系统油位，油温超过 65℃ 必须停机降温

## 四、新手质检员口诀
"一看二敲三过仪，车底车角不放过；单据照片留齐全，有异议时好说话。"
""",
}


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list:
    """简易分段：按段落切，每段尽量不超 size（Dify 的 General 模式也是这个思路）"""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n" + p).strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def embed(text: str) -> list:
    r = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {EMBED_KEY}"},
        json={"model": EMBED_MODEL, "input": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def main():
    client = QdrantClient(url=QDRANT_URL)
    # 建 collection：1024维 = bge-m3 的维度，COSINE = 余弦相似度
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print(f"[建库] {COLLECTION} 已创建（1024维）")

    points = []
    for name, text in DOCS.items():
        chunks = chunk_text(text)
        print(f"[切分] {name} → {len(chunks)} 段")
        for i, ch in enumerate(chunks):
            vec = embed(ch)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={"text": ch, "source": name},
                )
            )
            print(f"  向量化 {i + 1}/{len(chunks)}")

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"[完成] 共灌入 {len(points)} 个向量")


if __name__ == "__main__":
    main()
