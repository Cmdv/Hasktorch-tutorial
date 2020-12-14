{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import           Control.Monad (when)
import           GHC.Generics
import           Prelude hiding (exp)


-- import           Control.Monad ((<=<))
-- import           Control.Monad (forM_)
-- import           Control.Monad (forever)
import           Control.Monad.Cont (ContT(runContT))

import           Pipes
import qualified Pipes.Prelude as P

import           Torch
import qualified Torch.Functional as TF
import qualified Torch.Vision as V
-- import           Torch.Serialize
-- import           Torch.Data.Pipeline
import           Torch.Typed.Vision (initMnist)
import           Torch.Typed.Aux (StandardFloatingPointDTypeValidation)
import Data.IntMap.Internal (Nat)

data MLPSpec = MLPSpec
  { inputFeatures :: Int
  , hiddenFeatures0 :: Int
  , hiddenFeatures1 :: Int
  , outputFeatures :: Int
  } deriving (Show, Eq)

data MLP = MLP
  { layer0 :: Linear
  , layer1 :: Linear
  , layer2 :: Linear
  } deriving (Show, Generic, Parameterized)


instance Randomizable MLPSpec MLP where
    sample MLPSpec {..} = MLP
        <$> sample (LinearSpec inputFeatures hiddenFeatures0)
        <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
        <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

mlp :: MLP -> Tensor -> Tensor
mlp MLP{..} =
    logSoftmax (Dim 1)
    . linear layer2
    . relu
    . linear layer1
    . relu
    . linear layer0

instance HasForward MLP Tensor Tensor where
  forward MLP {..} = forward layer2 . TF.tanh . forward layer1 . TF.tanh . forward layer0
  forwardStoch = (pure .) . forward

trainLoop :: Optimizer o => MLP -> o -> ListT IO ((Tensor, Tensor), Int) -> IO  MLP
trainLoop model optimizer = P.foldM step begin done . enumerate
  where step :: MLP -> ((Tensor, Tensor), Int) -> IO MLP
        step model ((input, label), iter) = do
          let loss = nllLoss' label $ mlp model input
          when (iter `mod` 50 == 0) $ do
            putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
          (newParam, _) <- runStep model optimizer loss 1e-3
          pure newParam
        done = pure
        begin = pure model

displayImages :: MLP -> (Tensor, Tensor) -> IO ()
displayImages model (testImg, testLabel) =  do
  V.dispImage testImg
  putStrLn $ "Model        : " ++ (show . argmax (Dim 1) RemoveDim . exp $ mlp model testImg)
  putStrLn $ "Ground Truth : " ++ show testLabel

main :: IO ()
main = do
    (trainData, testData) <- initMnist "./data"
    let trainMnist = V.MNIST { batchSize = 64 , mnistData = trainData}
        testMnist = V.MNIST { batchSize = 1 , mnistData = testData}
        spec = MLPSpec 784 64 32 10
        optimizer = GD
    init <- sample spec
    trainedModel <- foldLoop init 5 $ \model _ -> do

      let tl = trainLoop model optimizer
          -- no instance for: Datastream m1 () (MNIST m0) (Tensor, Tensor)
          datasource = streamFrom' datastreamOpts trainMnist [()]

      pure model
      -- runContT (streamFromMap (datasetOpts 2) trainMnist) $ trainLoop model optimizer . fst

    -- show test images + labels
    -- forM_ [0..10]  $ displayImages trainedModel <=< getItem testMnist

    putStrLn "Done"
-- ListT IO (Tensor, Tensor) -> IO MLP
-- ListT IO ((Tensor, Tensor), Int) -> IO MLP
