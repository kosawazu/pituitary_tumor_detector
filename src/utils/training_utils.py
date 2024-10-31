

#EarlyStopping
class EarlyStopping:
    def __init__(
        self, 
        patience: int=10, 
        verbose: int=0
        ) -> None:
        '''
        Parameters:
            patience(int): 監視するエポック数(デフォルトは10)
            verbose(int): 早期終了の出力フラグ
                          出力(1),出力しない(0)        
        '''

        self.count = 0 # 監視中のエポック数のカウンターを初期
        self.pre_loss = float('inf') # 比較対象の損失を無限大'inf'で初期化
        self.patience = patience # 監視対象のエポック数をパラメーターで初期化
        self.verbose = verbose # 早期終了メッセージの出力フラグをパラメーターで初期化
        
    def __call__(
        self, 
        current_loss: float
        ) -> bool:
        '''
        Parameters:
            current_loss(float): 1エポック終了後の検証データの損失
        Return:
            True:監視回数の上限までに前エポックの損失を超えた場合
            False:監視回数の上限までに前エポックの損失を超えない場合
        '''
        
        if self.pre_loss < current_loss: # 前エポックの損失より大きくなった場合
            self.count += 1 # カウンターを1増やす

            if self.count > self.patience: # 監視回数の上限に達した場合
                if self.verbose:  # 早期終了のフラグが1の場合
                    print('early stopping')
                return True # 学習を終了するTrueを返す
            
        else: # 前エポックの損失以下の場合
            self.count = 0 # カウンターを0に戻す
            self.pre_loss = current_loss # 損失の値を更新する
        
        return False