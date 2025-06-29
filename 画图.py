from graphviz import Digraph
import os

def generate_model_diagram():
    # 创建有向图
    dot = Digraph(
        name='MultimodalRumorDetector',
        filename='multimodal_rumor_detector',
        format='png',
        engine='dot',
        graph_attr={
            'label': '多模态谣言检测模型架构',
            'labelloc': 't',
            'fontsize': '20',
            'fontname': 'Microsoft YaHei'
        },
        node_attr={
            'shape': 'record',
            'style': 'filled',
            'fontname': 'Microsoft YaHei'
        }
    )
    
    # 输入层
    dot.node('input', 
             label='<<table border="0" cellborder="0" cellspacing="0" cellpadding="3">'
                   '<tr><td colspan="3" align="center" bgcolor="#ffcccc"><b>输入层</b></td></tr>'
                   '<tr><td port="text_in" align="left">原始文本</td>'
                   '<td port="sentiment_in" align="left">情感分析</td>'
                   '<td rowspan="2" port="images_in" align="left">图像输入 (最多5张)</td></tr>'
                   '<tr><td colspan="2" port="text_input" align="center">"Breaking news: UFO spotted!"</td></tr>'
                   '</table>>',
             fillcolor='#ffe6e6')
    
    # 文本处理分支
    with dot.subgraph(name='cluster_text') as c:
        c.attr(label='文本处理分支', 
               style='filled', 
               fillcolor='#e6f2ff',
               fontsize='16',
               fontname='Microsoft YaHei')
        
        c.node('text_pre', 
               label='<<table border="0" cellborder="0" cellspacing="0">'
                     '<tr><td><b>文本预处理</b></td></tr>'
                     '<tr><td>- 移除URL/提及</td></tr>'
                     '<tr><td>- 清理特殊字符</td></tr>'
                     '</table>>',
               fillcolor='#cce6ff')
        
        c.node('bert', 
               label='<<table border="0" cellborder="0" cellspacing="0">'
                     '<tr><td><b>BERT编码器</b></td></tr>'
                     '<tr><td>(DistilBERT模型)</td></tr>'
                     '<tr><td>768维输出</td></tr>'
                     '</table>>',
               fillcolor='#99ccff')
        
        c.node('bilstm', 
               label='<<table border="0" cellborder="0" cellspacing="0">'
                     '<tr><td><b>BiLSTM层</b></td></tr>'
                     '<tr><td>128单元双向</td></tr>'
                     '<tr><td>256维输出</td></tr>'
                     '</table>>',
               fillcolor='#66b3ff')
        
        c.node('text_fc', 
               label='<<table border="0" cellborder="0" cellspacing="0">'
                     '<tr><td><b>文本特征压缩</b></td></tr>'
                     '<tr><td>256→128维</td></tr>'
                     '<tr><td>ReLU + Dropout(0.5)</td></tr>'
                     '</table>>',
               fillcolor='#3399ff')
        
        # 文本分支连接
        dot.edge('input:text_input', 'text_pre', style='dashed')
        c.edges([('text_pre', 'bert'), ('bert', 'bilstm'), ('bilstm', 'text_fc')])
    
    # 情感特征
    dot.node('sentiment', 
             label='<<table border="0" cellborder="0" cellspacing="0" cellpadding="3">'
                   '<tr><td bgcolor="#fffae6"><b>情感特征提取</b></td></tr>'
                   '<tr><td align="left">VADER情感分析器</td></tr>'
                   '<tr><td>4维输出：neg/neu/pos/compound</td></tr>'
                   '</table>>',
             fillcolor='#fffae6')
    
    # 图像处理分支
    with dot.subgraph(name='cluster_image') as c:
        c.attr(label='图像处理分支', 
               style='filled', 
               fillcolor='#e6ffe6',
               fontsize='16',
               fontname='Microsoft YaHei')
        
        c.node('image_pre', 
               label='<<table border="0" cellborder="0" cellspacing="0">'
                     '<tr><td><b>图像预处理</b></td></tr>'
                     '<tr><td>224×224像素</td></tr>'
                     '<tr><td>RGB标准化</td></tr>'
                     '</table>>',
               fillcolor='#ccffcc')
        
        c.node('resnet', 
               label='<<table border="0" cellborder="0" cellspacing="0">'
                     '<tr><td><b>ResNet18</b></td></tr>'
                     '<tr><td>512维特征/每图</td></tr>'
                     '<tr><td>(移除分类层)</td></tr>'
                     '</table>>',
               fillcolor='#99ff99')
        
        c.node('avg_pool', 
               label='<<table border="0" cellborder="0" cellspacing="0">'
                     '<tr><td><b>多图像平均池化</b></td></tr>'
                     '<tr><td>生成综合图像特征</td></tr>'
                     '</table>>',
               fillcolor='#66ff66')
        
        c.node('img_fc', 
               label='<<table border="0" cellborder="0" cellspacing="0">'
                     '<tr><td><b>图像特征压缩</b></td></tr>'
                     '<tr><td>512→256维</td></tr>'
                     '<tr><td>ReLU + Dropout(0.5)</td></tr>'
                     '</table>>',
               fillcolor='#33cc33')
        
        # 图像分支连接
        dot.edge('input:images_in', 'image_pre', style='dashed')
        c.edges([('image_pre', 'resnet'), ('resnet', 'avg_pool'), ('avg_pool', 'img_fc')])
    
    # 融合层
    with dot.subgraph(name='cluster_fusion') as c:
        c.attr(label='特征融合层', 
               style='filled', 
               fillcolor='#f0e6ff',
               fontsize='16',
               fontname='Microsoft YaHei')
        
        c.node('concat1', 
               label='<<table border="0" cellborder="0" cellspacing="0">'
                     '<tr><td><b>特征拼接</b></td></tr>'
                     '<tr><td>文本128D + 图像256D</td></tr>'
                     '<tr><td>384维融合特征</td></tr>'
                     '</table>>',
               shape='diamond',
               fillcolor='#d9c2ff')
        
        c.node('fusion_fc', 
               label='<<table border="0" cellborder="0" cellspacing="0">'
                     '<tr><td><b>图文融合层</b></td></tr>'
                     '<tr><td>384→124维</td></tr>'
                     '<tr><td>ReLU + Dropout(0.5)</td></tr>'
                     '</table>>',
               fillcolor='#c2a3ff')
        
        c.node('concat2', 
               label='<<table border="0" cellborder="0" cellspacing="0">'
                     '<tr><td><b>情感特征融合</b></td></tr>'
                     '<tr><td>图文124D + 情感4D</td></tr>'
                     '<tr><td>128维融合特征</td></tr>'
                     '</table>>',
               shape='diamond',
               fillcolor='#a885ff')
        
        c.node('final_fc', 
               label='<<table border="0" cellborder="0" cellspacing="0">'
                     '<tr><td><b>最终特征处理</b></td></tr>'
                     '<tr><td>128维特征</td></tr>'
                     '<tr><td>ReLU + Dropout(0.5)</td></tr>'
                     '</table>>',
               fillcolor='#8c68d9')
        
        # 融合层连接
        c.edges([('concat1', 'fusion_fc'), ('fusion_fc', 'concat2'), ('concat2', 'final_fc')])
    
    # 输出层
    dot.node('output', 
             label='<<table border="0" cellborder="0" cellspacing="0" cellpadding="3">'
                   '<tr><td colspan="2" bgcolor="#ffcccc"><b>输出层</b></td></tr>'
                   '<tr><td align="left">分类全连接</td>'
                   '<td>128→2维</td></tr>'
                   '<tr><td align="left">Softmax激活</td>'
                   '<td>概率输出</td></tr>'
                   '<tr><td bgcolor="#ccffcc" align="center" colspan="2">真实概率</td></tr>'
                   '<tr><td bgcolor="#ff9999" align="center" colspan="2">虚假概率</td></tr>'
                   '</table>>',
             fillcolor='#ffcccc')
    
    # 跨分支连接
    dot.edge('text_fc', 'concat1')
    dot.edge('img_fc', 'concat1')
    dot.edge('sentiment', 'concat2')
    dot.edge('final_fc', 'output')
    
    # 渲染图
    dot.render(
        view=True,  # 自动打开图像
        cleanup=True,  # 清理中间文件
        format='png',
        directory=os.getcwd(),
    )
    
    return dot

if __name__ == '__main__':
    print("生成模型架构图...")
    diagram = generate_model_diagram()
    print(f"架构图已保存到: {diagram.filename}.png")
    print("生成完成!")