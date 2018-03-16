using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;
using System.Linq;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training;
using ConvNetSharp.Volume;

public class TestAI1_LocalBrain : MonoBehaviour
{
    [SerializeField]
    int actionNum = 6;

    private Net<double> commonNet;
    private Net<double> piNet;
    private Net<double> vNet;
    private Net<double> tNet;

    private A3CAdamTrainer<double> c_trainer;
    private A3CAdamTrainer<double> p_trainer;
    private A3CAdamTrainer<double> v_trainer;
    private A3CAdamTrainer<double> t_trainer;

    private void Start()
    {
        //Network作る
        this.commonNet = new Net<double>();
        this.commonNet.AddLayer(new InputLayer(200, 200, 3));
        this.commonNet.AddLayer(new ConvLayer(5, 5, 6) { Stride = 1, Pad = 2 });
        this.commonNet.AddLayer(new ReluLayer());
        this.commonNet.AddLayer(new PoolLayer(4, 4) { Stride = 4 });
        this.commonNet.AddLayer(new ConvLayer(5, 5, 16) { Stride = 1, Pad = 2 });
        this.commonNet.AddLayer(new ReluLayer());
        this.commonNet.AddLayer(new PoolLayer(5, 5) { Stride = 5 });
        this.commonNet.AddLayer(new FullyConnLayer(50));
        c_trainer = new A3CAdamTrainer<double>(commonNet)
        {
            LearningRate = 0.01,
            BatchSize = 20,
            L2Decay = 0.001,
            Momentum = 0.9
        };

        this.piNet = new Net<double>();
        this.piNet.AddLayer(new InputLayer(50,1,1));
        this.piNet.AddLayer(new FullyConnLayer(25));
        this.piNet.AddLayer(new FullyConnLayer(actionNum));
        this.piNet.AddLayer(new SoftmaxLayer(actionNum));
        p_trainer = new A3CAdamTrainer<double>(commonNet)
        {
            LearningRate = 0.01,
            BatchSize = 20,
            L2Decay = 0.001,
            Momentum = 0.9
        };

        this.vNet = new Net<double>();
        this.vNet.AddLayer(new InputLayer(50,1,1));
        this.piNet.AddLayer(new FullyConnLayer(25));
        this.piNet.AddLayer(new FullyConnLayer(1));
        v_trainer = new A3CAdamTrainer<double>(commonNet)
        {
            LearningRate = 0.01,
            BatchSize = 20,
            L2Decay = 0.001,
            Momentum = 0.9
        };

        this.tNet = new Net<double>();
        this.tNet.AddLayer(new InputLayer(actionNum + 1, 1, 1));
        this.tNet.AddLayer(new CalcTotalLossLayer(1));
        t_trainer = new A3CAdamTrainer<double>(commonNet)
        {
            LearningRate = 0.01,
            BatchSize = 20,
            L2Decay = 0.001,
            Momentum = 0.9
        };
    }

    public void Train(Columun<double>[] clms)
    {
        //下降していく（inout1はsのreshape済み要素をbatch分並べる）
        Volume<T> input1;

        c_trainer.Forward(input1);//ここから、protectionLevel書き換えないとキツイ
        FullyConnLayer c_LastLayer = commonNet.Layers[commonNet.Layers.Count - 1];
        Volume<T> cOutA = c_LastLayer.OutputActivation;

        p_trainer.Forward(cOutA);
        FullyConnLayer p_LastLayer = piNet.Layers[piNet.Layers.Count - 1];
        Volume<T> pOutA = p_LastLayer.OutputActivation;

        v_trainer.Forward(cOutA);
        FullyConnLayer v_LastLayer = vNet.Layers[vNet.Layers.Count - 1];
        Volume<T> vOutA = v_LastLayer.OutputActivation;

        //pvOutAはpとvの出力を単純に結合したものです
        Volume<T> pvOutA;
        t_trainer.Forward(pvOutA);

        //逆伝播開始、Gradientを保存して、TrainImptemで一気に解決という流れ
        //input2はlossと逆伝播誤差を計算するのに必要なVolume。CalcTotalLossLayer参照
        Volume<T> input2;

        t_trainer.Backward(input2);
        CalcTotalLossLayer t_SecLayer = tNet.Layers[1];
        Volume<T> tOutAG = t_SecLayer.OutputActivationGradients;
        t_trainer.TrainImplem();

        v_trainer.Backward(tOutAG);
        FullyConnLayer v_SecLayer = vNet.Layers[1];
        Volume<T> vOutAG = v_SecLayer.OutputActivationGradients;
        v_trainer.TrainImplem();

        p_trainer.Backward(tOutAG);
        FullyConnLayer p_SecLayer = pNet.Layers[1];
        Volume<T> pOutAG = p_SecLayer.OutputActivationGradients;
        p_trainer.TrainImplem();

        Volume<T> pvOutAG = Ops<T>.Add(vOutAG, pOutAG);
        c_trainer.Backward(pvOutAG);
        c_trainer.TrainImplem();
    }
}

//行動と報酬を格納するクラス
public class Columun<T> where T : struct, IEquatable<T>, IFormattable
{
    public Volume<T> State { get; private set; }
    public Volume<T> Action { get; private set; }
    public float Reward { get; private set; }
    public Volume<T> State_ { get; private set; }

    public Columun(Volume<T> st, Volume<T> ac, float rew, Volume<T> st_)
    {
        State = st;
        Action = ac;
        Reward = rew;
        State_ = st_;
    }

    public Volume<T> AsVolume()
    {
        //Columunを1行の行列として取りだす()
        Volume<T> vol = State;

        return vol;
    }
}

//A3C特有のTrainを定義するクラ:変えなくてよさそうです(AdamTrainer使えばよい)
public class A3CAdamTrainer<T>: TrainerBase<T> where T : struct, IEquatable<T>, IFormattable
{
    private readonly List<Volume<T>> gsum = new List<Volume<T>>(); // last iteration gradients (used for momentum calculations)
    private readonly List<Volume<T>> xsum = new List<Volume<T>>();
    private int k;

    public A3CAdamTrainer(INet<T> net) : base(net)
    {
        if (typeof(T) == typeof(double))
        {
            this.Eps = (T)(ValueType)1e-8;
        }
        else if (typeof(T) == typeof(float))
        {
            this.Eps = (T)(ValueType)(float)1e-8;
        }
    }

    public T Beta1 { get; set; }

    public T Beta2 { get; set; }

    public T L1Decay { get; set; }

    public T L2Decay { get; set; }

    public T L2DecayLoss { get; private set; }

    public T L1DecayLoss { get; private set; }

    public T LearningRate { get; set; }

    public T Eps { get; set; }

    protected override void Backward(Volume<T> y)
    {
        base.Backward(y);

        this.L2DecayLoss = Ops<T>.Zero;//new T と同じ
        this.L1DecayLoss = Ops<T>.Zero;
    }

    protected override void TrainImplem()
    {
        var parametersAndGradients = this.Net.GetParametersAndGradients();

        // initialize lists for accumulators. Will only be done once on first iteration
        if (this.gsum.Count == 0)
        {
            foreach (var t in parametersAndGradients)
            {
                this.gsum.Add(BuilderInstance<T>.Volume.SameAs(t.Volume.Shape));
                this.xsum.Add(BuilderInstance<T>.Volume.SameAs(t.Volume.Shape));
            }
        }

        var factor = Ops<T>.Divide(Ops<T>.One, Ops<T>.Cast(this.BatchSize));

        // perform an update for all sets of weights
        for (var i = 0; i < parametersAndGradients.Count; i++)
        {
            var parametersAndGradient = parametersAndGradients[i];
            var vol = parametersAndGradient.Volume;
            var grad = parametersAndGradient.Gradient;

            // learning rate for some parameters.
            var l2DecayMul = parametersAndGradient.L2DecayMul ?? Ops<T>.One;
            var l1DecayMul = parametersAndGradient.L1DecayMul ?? Ops<T>.One;
            var l2Decay = Ops<T>.Multiply(this.L2Decay, l2DecayMul);
            var l1Decay = Ops<T>.Multiply(this.L1Decay, l1DecayMul);

            //  this.L2DecayLoss += l2Decay * vol.Get(j) * vol.Get(j) / 2; // accumulate weight decay loss
            //  this.L1DecayLoss += l1Decay * Math.Abs(vol.Get(j));

            var l1Grad = vol.Clone();
            l1Grad.MapInplace(x => Ops<T>.GreaterThan(x, Ops<T>.Zero) ? Ops<T>.One : Ops<T>.Negate(Ops<T>.One));
            l1Grad = l1Grad * l1Decay;

            var l2Grad = vol * l2Decay;

            var gij = (grad + l2Grad + l1Grad) * factor;

            // momentum update
            this.gsum[i] = this.gsum[i] * this.Beta1 + gij * Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(this.Beta1)); // update biased first moment estimate
            var gijgij = gij.Clone();
            gijgij.MapInplace(x => Ops<T>.Multiply(x, x));
            this.xsum[i] = this.xsum[i] * this.Beta2 + gijgij * Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(this.Beta2)); // update biased second moment estimate
            var biasCorr1 = this.gsum[i] * Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(Ops<T>.Pow(this.Beta1, Ops<T>.Cast(this.k)))); // correct bias first moment estimate
            var biasCorr2 = this.xsum[i] * Ops<T>.Add(Ops<T>.One, Ops<T>.Negate(Ops<T>.Pow(this.Beta2, Ops<T>.Cast(this.k)))); // correct bias second moment estimate
            biasCorr2.MapInplace(x => Ops<T>.Add(Ops<T>.Sqrt(x), this.Eps));

            var dx = biasCorr1 * this.LearningRate;
            dx.MapInplace((l, r) => Ops<T>.Divide(l, r), biasCorr2);

            vol.MapInplace((v, d) => d, vol - dx); // apply corrected gradient

            grad.Clear(); // zero out gradient so that we can begin accumulating anew
        }

        this.k += this.BatchSize;
    }
}

//total loss計算して逆伝播開始するレイヤー。
//入力がふたつある点でエラーの原因になるかも
namespace ConvNetSharp.Core.Layers
{
    public class CalcTotalLossLayer<T> : LastLayerBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public CalcTotalLossLayer(int cCount)
        {
            this.ClassCount = cCount;
        }

        public int ClassCount { get; set; }

        public override void Backward(Volume<T> y, out T loss) //y入力は1行行列にしたColumunで
        {
            //OutputActivationGradientに誤差を入れる

            //上部InputLayerの出力とyでlossを計算する。
            loss = Ops<T>.Zero;

            loss = Ops<T>.Negate(loss);

            if (Ops<T>.IsInvalid(loss))
                throw new ArgumentException("Error during calculation!");
        }

        public override void Backward(Volume<T> outputGradient)
        {
            throw new NotImplementedException();
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            //InputActivationから計算出来るものは計算してしまう
            return this.OutputActivation;
        }

        public override Dictionary<string, object> GetData()
        {
            var dico = base.GetData();
            dico["ClassCount"] = this.ClassCount;
            return dico;
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            var inputCount = inputWidth * inputHeight * inputDepth;
            this.OutputWidth = 1;
            this.OutputHeight = 1;
            this.OutputDepth = inputCount;
        }
    }
}