# waf-test
CSIC 2010 HTTP DATASETを用いて機械学習して攻撃検知する

## Notes
- anomalousTrafficTest.txtとnormalTrafficTraining.txtを用いて10分割交差検証
- アルゴリズムはランダムフォレスト
- HTTPリクエスト中のパラメータをASCIIコード中の記号で分割し，トークン配列に変換
- Accuracy: 約99.8%
