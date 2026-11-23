# 参考文献・出典

1. Chatrath, V., Zhu, B., Fan, J., et al. (2026). *READY or Not: Reliable Enterprise Agent Deployment*. arXiv:2609.02095 v1. https://arxiv.org/abs/2609.02095 （CC BY 4.0）
2. Scale Labs. (2026-09-03). *READY*（公式解説ブログ）. https://labs.scale.com/blog/ready
3. Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., & Vulić, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders*. Proceedings of the 2nd Workshop on NLP for Conversational AI. arXiv:2003.04807 （BANKING77 の原典）
4. PolyAI. *banking77*（データセット、CC BY 4.0）. https://huggingface.co/datasets/PolyAI/banking77
5. Anthropic. Claude API ドキュメント（モデル一覧・料金・構造化出力）. https://platform.claude.com/docs

## データセットの帰属表示

本実験は PolyAI が CC BY 4.0 で公開している BANKING77 を使用した。データは改変していない（抽出のみ）。

## 用語

- 選択的予測（selective prediction）・リスク・カバレッジ曲線・AURC：確信度の高いものだけをAIに任せ、残りを保留する考え方とその評価指標。READY 論文はこれを導入判断の手続きに組み込んでいる。
- Clopper–Pearson 区間：二項比率の正確な信頼区間。本実験では片側下限のみ使用。
