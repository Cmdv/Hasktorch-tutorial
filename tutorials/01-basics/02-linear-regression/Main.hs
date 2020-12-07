module Main where

import Control.Monad (when)
import Torch.Optim (GD(..), runStep, foldLoop)
import Torch.Tensor (asTensor, Tensor, select, size)
import Torch.Device (Device(..), DeviceType(..))
import Torch.NN (linear, sample, LinearSpec(..), Linear(..))
import Torch.Functional (mseLoss, matmul, squeezeAll, view)
import Torch.TensorFactories (full')
import Torch.Random (mkGenerator, randn')
import Torch (IndependentTensor(..))
import qualified Graphics.Vega.VegaLite as V
import Torch.Typed.NN (HasForward(forward))
import Control.Monad.Trans.Cont
import Torch.Typed.Tensor (toFloat)
import Control.Monad.Trans.Class


{-
 ==================================================================
                         Table of Contents
 ==================================================================

 1. Basic Tensor                           (Line 35  to 89)
 1. N Dimensional Tensors                  (Line 35  to 89)
 1. Differentiation                        (Line 35  to 89)
 1. Basic autograd example 1               (Line 31  to 47)
 2. Basic autograd example 2               (Line 49  to __)
 3. Loading data from numpy                (Line __  to __)
 4. Input pipline                          (Line __  to __)
 5. Input pipline for custom dataset       (Line __  to __)
 6. Pretrained model                       (Line __  to __)
 7. Save and load model                    (Line __  to __)

 ==================================================================
                1. Linear Regression Prediction 1D
 ==================================================================
-}
linearRPrediction1D :: IO ()
linearRPrediction1D = do
  let w = asTensor (2.0 :: Float)
      b = asTensor (-1.0 :: Float)
      x = asTensor ([[1.0],[2.0]] :: [[Float]])
      -- custom linear function
      -- y' = b + w x
      y = b + w * x
  printTensor "1: Linear function" y
  -- create a model
  model <- sample $ LinearSpec { in_features = 1, out_features = 1 }
  printTensor "2: Linear weight & bias" model
  printTensor "3: Weight" $ weight model
  printTensor "4: Bias" $ bias model

  -- pass x to model
  let result = forward model x
  printTensor "4:Run Model against x" result

{-
 ==================================================================
                2. Linear Regression Prediction 1D
 ==================================================================
-}

xTrain :: [[Float]]
xTrain =
  [ [3.3], [4.4], [5.5], [6.71], [6.93], [4.168]
  , [9.779], [6.182], [7.59], [2.167], [7.042]
  , [10.791], [5.313], [7.997], [3.1]
  ]

yTrain :: [[Float]]
yTrain =
  [ [1.7], [2.76], [2.09], [3.19], [1.694], [1.573]
  , [3.366], [2.596], [2.53], [1.221], [2.827]
  , [3.465], [1.65], [2.904], [1.3]
  ]

linearRegression :: IO ()
linearRegression = do
  -- convert lists to torch tensors
  let inputs = asTensor xTrain
      targets = asTensor yTrain

  -- set up our model
  model <- sample $ LinearSpec { in_features = inputSize, out_features = outputSize }

  -- our loop
  trainedPrediction <- foldLoop model numEpochs $ \modelState i -> do
        -- combine out inputs to our modelState
    let expectedOutputs = squeezeAll $ linear modelState inputs
        -- change the shape of the vector from [n,1] -> [n]
        targets' = view [-1] targets
        -- do a mean squared errors on
        loss = mseLoss targets' expectedOutputs
    -- every 100 iteratiosn print out the loss
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Loss': " ++ show loss
    -- run our optimizer steps and backwards loss
    (newModel, _) <- runStep modelState optimizer loss learningRate
    pure newModel
  -- print out the weight and bias
  putStrLn $ "Weight:\n" ++ show (toDependent $ weight trainedPrediction)
  putStrLn $ "Bias:\n" ++ show (toDependent $ bias trainedPrediction)
  pure ()
  where
    -- Parameters
    inputSize = 1
    outputSize = 1
    numEpochs = 3000
    learningRate = 1e-5
    optimizer = GD


{-
 ==================================================================
                3.
 ==================================================================
-}



printTensor :: Show a => String -> a -> IO()
printTensor s t = do
  putStr $ "---------- " <> s  <> " ---------- " <> "\n" <> show t <> "\n\n"


main :: IO ()
main = do
  linearRegression
