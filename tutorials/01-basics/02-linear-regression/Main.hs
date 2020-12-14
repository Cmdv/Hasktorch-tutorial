module Main where

import Control.Monad (when)
import Torch.Optim (GD(..), runStep, foldLoop)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.NN (linear, sample, LinearSpec(..))
import Torch.Functional (mseLoss, view)
import qualified Graphics.Vega.VegaLite as V
import GHC.Float (float2Double)

{-
 ==================================================================
                1. Linear Regression 1D
 ==================================================================

-- We are trying to solve a problem of finding a house value given it's size
-- in m2.
-- We have an existing data set of size of houses and their values.
-- We will use the training sets to teach our model.

-}

-- house size training set
xTrain :: [[Float]]
xTrain =
  [ [1700.0], [2760.0], [2090.0], [3190.0], [1694.0], [1573.0]
  , [3366.0], [2596.0], [2530.0], [1221.0], [2827.0]
  , [3465.0], [1650.0], [2904.0], [1300.0]
  ]

-- house value training set
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
    when (i `mod` 10 == 0) $ do
      putStrLn $ "Epoch Num:" ++ show i ++ " Loss': " ++ show loss
    -- run our gradient decent optimizer and backwards loss
    (newModel, _) <- runStep modelState optimizer loss learningRate
    -- return the current state of the training model to be picked up by next epoch itteration.
    pure newModel
  -- show use of trainedModel working against our training set. The result can be used to draw the linear regression line
  -- in a chart.
  -- putStrLn $ "\n Values to draw regression line:\n " ++ show (linear trainedModel inputs)
  -- display chart
  putStrLn $ show trainedModel
  let trainedData = linear trainedModel inputs
      trainedList = asValue (view [-1] trainedData) :: [Float]
  plot "linearReg.html" trainedList
  pure ()
  where
    -- Parameters
    inputSize = 1
    outputSize = 1
    numEpochs = 40
    learningRate = 1e-8
    optimizer = GD

plot :: FilePath -> [Float] -> IO ()
plot file trainedY = do
  let w = V.width 700
      h = V.height 600

      manualData = V.dataFromColumns []
                 . V.dataColumn "House size" (V.Numbers xData)
                 . V.dataColumn "House Price" (V.Numbers yData)
                 . V.dataColumn "Estimated Price" (V.Numbers $ float2Double <$> trainedY)
                 $ []

      xData = map float2Double $ concat xTrain
      yData = map float2Double $concat yTrain
      encDots = V.encoding
          . V.position V.X [V.PName "House size", V.PmType V.Quantitative]
          . V.position V.Y [V.PName "House Price", V.PmType V.Quantitative]
      enc = V.encoding
          . V.position V.X [V.PName "House size", V.PmType V.Quantitative]
          . V.position V.Y [V.PName "Estimated Price", V.PmType V.Quantitative]

  V.toHtmlFile file $ V.toVegaLite
   [ manualData
   , V.layer [ V.asSpec [ encDots []
                        , V.mark V.Point [V.MSize 5, V.MStroke "black"]
                        ]
             , V.asSpec [ enc []
                        , V.mark V.Line [ V.MStroke "red"
                                        , V.MStrokeWidth 3]
                        ]
             ]
   , w
   , h
   ]

main :: IO ()
main = do
  linearRegression
