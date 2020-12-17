{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import           Control.Monad (when)
import           GHC.Generics
import           Prelude hiding (exp)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BSL
import           Control.Monad ((<=<), forM_)
import           Control.Monad.Cont (ContT(..))
import qualified Pipes.Prelude as P
import           Pipes
import           Torch
import qualified Torch.Vision as V
import qualified Torch.Typed.Vision as TV

data MLPSpec = MLPSpec {
    inputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int,
    outputFeatures :: Int
    } deriving (Show, Eq)

data MLP = MLP {
    l0 :: Linear,
    l1 :: Linear,
    l2 :: Linear
    } deriving (Generic, Show)

instance Parameterized MLP
instance Randomizable MLPSpec MLP where
    sample MLPSpec {..} = MLP
        <$> sample (LinearSpec inputFeatures hiddenFeatures0)
        <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
        <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

mlp :: MLP -> Tensor -> Tensor
mlp MLP{..} =
    logSoftmax (Dim 1)
    . linear l2
    . relu
    . linear l1
    . relu
    . linear l0

trainLoop :: Optimizer a => MLP -> a -> ListT IO (Tensor, Tensor) -> IO  MLP
trainLoop model optimizer = P.foldM step (pure model) pure . enumerateData
  where
    step :: MLP -> ((Tensor, Tensor), Int) -> IO MLP
    step curModel ((input, label), iter) = do
      let loss = nllLoss' label $ mlp curModel input
      when (iter `mod` 50 == 0) $ do
        putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
      (newParam, _) <- runStep curModel optimizer loss 1e-3
      pure newParam

main :: IO ()
main = do
    (trainData, testData) <- initMnistFiles "data"
    let trainMnist = V.MNIST { batchSize = 32 , mnistData = trainData}
        testMnist = V.MNIST { batchSize = 1 , mnistData = testData}
        spec = MLPSpec 784 64 32 10
        optimizer = GD
    init <- sample spec
    model <- foldLoop init 5 $ \model _ ->
      runContT (streamFromMap (datasetOpts 2) trainMnist) $ trainLoop model optimizer . fst

    -- show test images + labels
    forM_ [0..10]  $ displayImages model <=< getItem testMnist

    putStrLn "Done"



{-
 ==================================================================
                     2. Helper functions
 ==================================================================
-}

displayImages :: MLP -> (Tensor, Tensor) -> IO ()
displayImages model (testImg, testLabel) =  do
  V.dispImage testImg
  putStrLn $ "Model        : " ++ (show . argmax (Dim 1) RemoveDim . exp $ mlp model testImg)
  putStrLn $ "Ground Truth : " ++ show testLabel


filetoBS :: String -> String -> IO BS.ByteString
filetoBS path file = go <$> BS.readFile (path <> "/" <> file)
  where
    go = BS.concat . BSL.toChunks . BSL.fromStrict

initMnistFiles :: String -> IO (TV.MnistData, TV.MnistData)
initMnistFiles path = do
  imagesBS <- filetoBS path "train-images-idx3-ubyte"
  labelsBS <- filetoBS path "train-labels-idx1-ubyte"
  testImagesBS <- filetoBS path "t10k-images-idx3-ubyte"
  testLabelsBS <- filetoBS path "t10k-labels-idx1-ubyte"
  return (TV.MnistData imagesBS labelsBS, TV.MnistData testImagesBS testLabelsBS)
