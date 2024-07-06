## README for BioQwen Model Deployment on Mobile Devices

### Overview

The BioQwen model, developed under the MLC-LLM project, showcases advanced capabilities in handling medical queries through a compact, mobile-friendly design. Utilizing INT4 model compression technology, we have successfully reduced the size of BioQwen 0.5B to under 300MB and BioQwen 1.8B to under 1GB. This enables seamless deployment on mobile devices, balancing storage and computational constraints.

### Model Performance

BioQwen's performance, particularly in handling complex medical queries, was evaluated using the cMedQA2 dataset. This dataset poses more challenging queries than WebMedQA, providing a robust test environment to assess the model's practical application in real-world scenarios.

### Deployment and Testing

Post-deployment, the BioQwen model was subjected to practical application tests on mobile devices. The results, particularly in response to specific medical inquiries, demonstrate the model's accuracy, richness in information, and empathetic response quality. These attributes highlight BioQwen's reliability and effectiveness in real-world use cases.

### Example Responses

Two illustrative examples of the BioQwen model's responses are provided:

1. **HIV Infection Risk Post High-Risk Behavior**:
    - **Original Question (Chinese)**: 
        - 用户描述了两年前经历的高危行为后进行的HIV检测，结果为阴性，但随后出现了咽炎和轻微感冒，担心是否会感染HIV，并询问是否需要再次检测。
    - **Model Response (Translated to English)**:
        - **Question**: "A while ago, I suddenly remembered a high-risk behavior from two years ago. I went for an HIV test, and the result was negative. However, a week later, I developed a sore throat and slight cold. I'm worried if I could have contracted HIV. Should I get tested again?"
        - **Response**: "If your test is negative and you are not engaging in high-risk behaviors, your risk of infection is very low. Stress or allergies might have caused your symptoms. Seek medical attention for further evaluation and maintain good health practices to boost your immunity."

2. **Pregnancy Symptoms and Checkup Frequency**:
    - **Original Question (Chinese)**:
        - 用户描述了怀孕后期左腿抽筋的症状，询问是否需要补钙或进行更多检查，并担心对日常生活和工作有无影响。
    - **Model Response (Translated to English)**:
        - **Question**: "After the third trimester of pregnancy, my left leg hurts like a cramp, and sometimes it's so painful that I can't walk. Is it due to calcium deficiency? Should we go for more checkups? My wife is pregnant with twins; will it affect her daily life and work?"
        - **Response**: "You may be experiencing common pregnancy symptoms. These usually occur early in pregnancy and may last for some time. If pain is severe, rest and apply warm compresses. For further assurance, consult your doctor. Regular monthly checkups are recommended to monitor the fetus's health."

### Conclusion

The BioQwen model, when deployed on mobile devices, offers robust, accurate, and compassionate responses to complex medical queries. Its successful deployment and testing on the cMedQA2 dataset demonstrate its practical utility and reliability in real-world applications. The model's compact size, made possible through INT4 compression, ensures it fits within the constraints of mobile device storage and computational capacities, making it an invaluable tool for accessible medical advice and information.

### Files Included
- **image1.png**: Screenshot of the BioQwen model's response to a high-risk behavior HIV query.
- **image2.png**: Screenshot of the BioQwen model's response to a pregnancy symptoms query.

### Contact
For any questions or further information, please submit an issue on this repository.
