{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import           GHC.Generics
import           Prelude hiding (exp)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BSL
import           Control.Monad ((<=<), forM_, when)
import           Control.Monad.Cont (ContT(..))
import qualified Pipes.Prelude as P
import           Pipes
import           Torch
import qualified Torch.Vision as V
import qualified Torch.Typed.Vision as TV

{-
  This is a representation of our neural network
  It has an input layer, 2 hidden layers and an output

  +---------+    +---------+    +---------+    +---------+
  |  Input  +--> | Hidden  +--> | Hidden  +--> | Output  |
  |         |    | layer1  |    | layert2 |    |         |
  +---------+    +---------+    +---------+    +---------+
-}

data MLP = MLP {
    hiddenL0 :: Linear,
    hiddenL1 :: Linear,
    outputL  :: Linear
    } deriving (Generic, Show)

data MLPSpec = MLPSpec {
    inputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int,
    outputFeatures :: Int
    } deriving (Show, Eq)

instance Parameterized MLP

-- we want to be able to create some random MLPSpecs
instance Randomizable MLPSpec MLP where
    sample MLPSpec {..} = MLP
        <$> sample (LinearSpec inputFeatures hiddenFeatures0)
        <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
        <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

-- here is the flow the input tensor takes through each hidden layer
-- and output layer.
-- Then it is all converted to a 1 dementional Tensor using a logSoftmax
mlp :: MLP -> Tensor -> Tensor
mlp MLP{..} input =
    logSoftmax (Dim 1)
    . linear outputL
    . relu
    . linear hiddenL1
    . relu
    . linear hiddenL0 $ input

-- The training loop
trainLoop :: Optimizer a => MLP -> a -> ListT IO (Tensor, Tensor) -> IO  MLP
trainLoop model optimizer =
  --
  P.foldM step (pure model) pure . enumerateData
  where
    step :: MLP -> ((Tensor, Tensor), Int) -> IO MLP
    step curModel ((input, label), iter) = do
      let loss = nllLoss' label $ mlp curModel input
      when (iter `mod` 50 == 0) $ do
        putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
      (newParam, _) <- runStep curModel optimizer loss 1e-3
      pure newParam

-- The main setup and loop
main :: IO ()
main = do
  -- make sure to run data/dowload_data.sh in route of repo
  -- this part will run agains the files generated and convert them
  -- into MnistData, see initMnistFiles in hellper functions bellow.
  (trainData, testData) <- initMnistFiles "data"

  -- setup MNIST which we can give a batch size attribute, which become useful
  -- when dealing with very large data sets.
  let trainMnist = V.MNIST { batchSize = 32 , mnistData = trainData}
      testMnist = V.MNIST { batchSize = 1 , mnistData = testData}
      -- This is the shape of our neural network
      -- (28 x 28) 784 inputs,
      -- 64 neurons in the first hidden layer
      -- 32 nearons in the second layer
      -- 10 output neurons for the numbers [0..9]
      spec = MLPSpec 784 64 32 10
      -- use th Gradient decent optimizer
      optimizer = GD
  -- setup a random sample for our neural network, to get out initial
  -- biases and weights
  init <- sample spec

  -- here is out initial loop which we run 5 times
  trainedModel <- foldLoop init 5 $ \model _ ->
    -- because that learning data is in batches we want to run thing using the continuation monad
    -- we streamFrom map our training data set and pass the result of that to trainloop
    -- runContT will return :: IO MLP
    runContT (streamFromMap (datasetOpts 2) trainMnist) (trainLoop model optimizer . fst)

  -- show test images + labels
  forM_ [0..10]  $ displayImages trainedModel <=< getItem testMnist

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
