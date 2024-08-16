import streamlit as st
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import os
import pandas as pd
from datetime import datetime

def run():
    # Page configuration
    st.set_page_config(
        page_title="Brain Tumor Classificatitor",
        page_icon="🧠",
        layout="wide"
    )
    
    # Custom CSS for hospital-like styling
    st.markdown(
        """
        <style>
        body {
            background-color: #f8f9fa;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #003366;
            font-family: 'Helvetica', sans-serif;
        }
        .reportview-container .main .block-container {
            padding: 2rem 1rem 1rem 1rem;
        }
        .sidebar .sidebar-content {
            background-color: #e3f2fd;
        }
        .sidebar .sidebar-content .css-1e5imcs {
            color: #003366;
        }
        .stImage {
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .custom-paragraph {
            font-size: 16px;
            line-height: 1.6;
            text-align: justify;
            margin-bottom: 20px;
        }
        .section-separator {
            border-top: 2px solid #003366;
            margin: 40px 0;
        }
        .footer {
            text-align: center;
            padding: 10px;
            background-color: #003366;
            color: white;
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.markdown("<h1 style='text-align: center;'>ỨNG DỤNG CÔNG NGHỆ AI TRONG VIỆC PHÁT HIỆN VÀ PHÂN LOẠI KHỐI U NÃO</h1>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>Thông tin hướng dẫn</h1>", unsafe_allow_html=True)
        st.sidebar.info("Website hiện chỉ có thể chuẩn đoán dựa trên ảnh chụp MRI. Bạn nên đến cơ sở y tế gần nhất để chụp ảnh MRI trước khi chxẩn đoán nhé.")
        st.markdown(
            """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 330px;
           max-width: 330px;
        }
        """,
            unsafe_allow_html=True,
        )   

    # Section: Current Situation
    st.markdown("<div class='section-separator'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Hiện trạng</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('''<h2 style='text-align: left; color: blue;'>
                </h2>''',
                unsafe_allow_html=True)
        st.markdown(
            """
            <div class="custom-paragraph">
            Theo y khoa, <span style="color: blue;"><b>khối u não</b></span> là một bệnh lý cực kỳ nguy hiểm bởi nó ảnh hưởng trực tiếp đến chức năng của não bộ và hệ thần kinh với tỷ lệ tử vong cao. U não có thể xuất hiện ở bất kỳ độ tuổi nào, trong đó đặc biệt là ở nhóm người trên 70 tuổi và trẻ em dưới 15 tuổi. Việc chẩn đoán chính xác và điều trị kịp thời đóng vai trò quan trọng trong việc <span style="color: blue;"><b>ngăn chặn sự phát triển</b></span> của khối u và cứu sống bệnh nhân. <span style="color: blue;"><b>Ảnh chụp MRI</b></span> là phương pháp chẩn đoán được sử dụng trong việc phát hiện các khối u não. Tuy nhiên, việc chẩn đoán và phân tích các loại khối u não thông qua ảnh MRI đòi hỏi thời gian và chuyên môn cao từ các y bác sĩ, có thể dẫn đến tình trạng chẩn đoán sai, gây ra những hậu quả đáng tiếc và sự thường xuyên quá tải lên hệ thống y tế.
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.image('brain.jpeg', caption='Khối u não')

    # Section: MRI Technology
    st.markdown("<div class='section-separator'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Công nghệ MRI</h2>", unsafe_allow_html=True)
    col3, col4 = st.columns([1, 2])
    with col3:
        st.image('info2.jpeg', caption='Máy chụp MRI')
    with col4:
        st.markdown('''<h4 style='text-align: left; color: blue;'>
                </h4>''',
                unsafe_allow_html=True)
        st.markdown(
            """
            <div class="custom-paragraph">
            <span style="color: blue;"><b>MRI (Magnetic Resonance Imaging)</b></span> là một kỹ thuật chụp ảnh y khoa sử dụng từ trường mạnh và sóng vô tuyến để <span style="color: blue;"><b>tạo ra hình ảnh chi tiết</b></span> của các cơ quan và mô trong cơ thể. MRI không sử dụng tia X và được sử dụng rộng rãi để chẩn đoán và theo dõi nhiều loại bệnh lý khác nhau, bao gồm tổn thương não, cột sống, cơ xương và tim mạch.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Section: Common Brain Tumor Types
    st.markdown("<div class='section-separator'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Một số loại khối u não phổ biến hiện nay</h2>", unsafe_allow_html=True)
    col5, col6 = st.columns([1, 3])
    with col5:
        st.image('p.jpg', caption='U thần kinh đệm (Gliomas)')
    with col6:
        st.markdown('''<h1 style='text-align: left; color: blue;'>
                </h1>''',
                unsafe_allow_html=True)
        
        #glioma = read_markdown_file("glioma.md") 
        #st.markdown(glioma, unsafe_allow_html=True)

    col7, col8 = st.columns([1, 3])
    with col7:
        st.image('m.jpg', caption='U màng não (Meningioma)')
    with col8:
        st.markdown('''<h2 style='text-align: left; color: blue;'>
                </h2>''',
                unsafe_allow_html=True)
        st.markdown(
            """
            <div class="custom-paragraph">
            <span style="color: red;"><b>U màng não (meningioma)</b></span> là một loại u não thường lành tính, phát triển từ các màng bao quanh não và tủy sống. U này thường tiến triển chậm và có thể gây ra các triệu chứng như đau đầu, co giật, và các vấn đề về thị giác hoặc vận động tùy thuộc vào vị trí và kích thước của khối u.
            </div>
            """,
            unsafe_allow_html=True
        )

    col9, col10 = st.columns([1, 3])
    with col9:
        st.image('gg.jpg', caption='U tuyến yên (Pituitary)')
    with col10:
        st.markdown('''<h2 style='text-align: left; color: blue;'>
                </h2>''',
                unsafe_allow_html=True)
        st.markdown(
            """
            <div class="custom-paragraph">
            <span style="color: red;"><b>U tuyến yên (pituitary)</b></span> là một loại u thường lành tính phát triển trong tuyến yên, một tuyến nhỏ ở đáy não. U này có thể ảnh hưởng đến sự sản xuất hormone, gây ra các triệu chứng như rối loạn hormone, đau đầu, thay đổi thị giác và các vấn đề khác liên quan đến hệ nội tiết.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Footer
    st.markdown("<div class='section-separator'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="footer">
        © 2024 Brain Tumor Classificatitor - Tác giả: Trần An Nguyên, Phan Lê Quỳnh Như
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    run()