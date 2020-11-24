module Main where

import Control.Monad (when)
import Torch.Optim (GD(..), runStep, foldLoop)
import Torch.Tensor (asTensor, Tensor)
import Torch.Device (Device(..), DeviceType(..))
import Torch.NN (linear, sample, LinearSpec(..), Linear(..))
import Torch.Functional (mseLoss, matmul, squeezeAll)
import Torch.TensorFactories (full')
import Torch.Random (mkGenerator, randn')
import Torch (IndependentTensor(..))
import qualified Graphics.Vega.VegaLite as V

model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input

groundTruth :: Tensor -> Tensor
groundTruth t = squeezeAll $ matmul t weight + bias
  where
    weight = asTensor ([42.0, 64.0, 96.0] :: [Float])
    bias = full' [1] (3.14 :: Float)

printParams :: Linear -> IO ()
printParams trained = do
    putStrLn $ "Parameters:\n" ++ show (toDependent $ weight trained)
    putStrLn $ "Bias:\n" ++ show (toDependent $ bias trained)

main :: IO ()
main = do
    init <- sample $ LinearSpec { in_features = numFeatures, out_features = 1 }
    randGen <- defaultRNG
    printParams init
    (trained, _) <- foldLoop (init, randGen) numIters $ \(state, randG) i -> do
        let (rdnInput, randGen') = randn' [batchSize, numFeatures] randG
            (y, y') = (groundTruth rdnInput, model state rdnInput)
            loss = mseLoss y y'

        when (i `mod` 100 == 0) $ do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
        (newParam, _) <- runStep state optimizer loss 5e-3
        pure (newParam, randGen')
    printParams trained
    pure ()
  where
    optimizer = GD
    defaultRNG = mkGenerator (Device CPU 0) 31415
    batchSize = 4
    numIters = 2000
    numFeatures = 3



mkVegaLite :: V.Data -> V.VegaLite
mkVegaLite dataset =
  let w = V.width 700
      h = V.height 350

      encOverview =
        V.encoding
          . V.position V.X [V.PName "x", V.PmType V.Quantitative]
          . V.position V.Y [V.PName "y", V.PmType V.Quantitative, V.PScale scaleOptsOverview]
      scaleOptsOverview = [V.SDomain (V.DNumbers [0, 10]), V.SNice (V.IsNice False)]
      transOverview =
        V.transform
          . V.filter (V.FRange "x" (V.NumberRange 0 20))
          . V.filter (V.FRange "y" (V.NumberRange 0 10))
      target =
        V.asSpec
          [ V.dataFromSource "target" [],
            transOverview [],
            V.mark V.Line [V.MStrokeWidth 0.5, V.MStroke "red"]
          ]
      evaluation =
        V.asSpec
          [ V.dataFromSource "evaluation" [],
            transOverview [],
            V.mark V.Point [V.MSize 5, V.MStroke "black"]
          ]
      prediction =
        V.asSpec
          [ V.dataFromSource "prediction" [],
            transOverview [],
            V.mark V.Line []
          ]
      overview =
        V.asSpec
          [ V.layer [target, evaluation, prediction],
            encOverview [],
            w,
            h
          ]

   in V.toVegaLite
        [ dataset,
          V.vConcat [overview]
        ]
