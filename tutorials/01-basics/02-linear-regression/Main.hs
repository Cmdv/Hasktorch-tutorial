module Main where

import Control.Monad (when)
import Torch.Optim (GD(..), runStep, foldLoop)
import Torch.Tensor (asTensor)
import Torch.NN (linear, sample, LinearSpec(..))
import Torch.Functional (mseLoss)

{-
 ==================================================================
                1. Linear Regression 1D
 ==================================================================
-}

-- the training sets for x and y axis
xTrain :: [[Float]]
xTrain =
  [ [1700.0], [2760.0], [2090.0], [3190.0], [1694.0], [1573.0]
  , [3366.0], [2596.0], [2530.0], [1221.0], [2827.0]
  , [3465.0], [1650.0], [2904.0], [1300.0]
  ]

yTrain :: [[Float]]
yTrain =
  [ [33000.0], [44000.0], [55000.0], [67100.0], [69300.0], [41680.0]
  , [97790.0], [61820.0], [75900.0], [21670.0], [70420.0]
  , [107910.0], [53130.0], [79970.0], [31000.0]
  ]

linearRegression :: IO ()
linearRegression = do
  -- convert training sets to torch tensors
  let inputs = asTensor xTrain
      targets = asTensor yTrain

  -- set up our random dummy model
  model <- sample $ LinearSpec { in_features = inputSize, out_features = outputSize }
  -- our loop
  trainedModel <- foldLoop model numEpochs $ \modelState i -> do
        -- combine out inputs to our modelState
    let expectedOutputs = linear modelState inputs
        -- do a mean squared errors on
        loss = mseLoss targets expectedOutputs
    -- every 50 iteratiosn print out the loss
    when (i `mod` 50 == 0) $ do
      putStrLn $ "Loss': " ++ show loss
    -- run our gradient decent optimizer and backwards loss
    (newModel, _) <- runStep modelState optimizer loss learningRate
    pure newModel
  -- show use of trainedModel working against our training set. The result can be used to draw the linear regression line
  -- in a chart.
  putStrLn $ "\n Values to draw regression line:\n " ++ show (linear trainedModel inputs)
  pure ()
  where
    -- Parameters
    inputSize = 1
    outputSize = 1
    numEpochs = 4000
    learningRate = 1e-10
    optimizer = GD


main :: IO ()
main = do
  linearRegression
