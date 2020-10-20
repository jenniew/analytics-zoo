package com.intel.analytics.zoo.tfpark

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.utils.{CachedModels, ModelBroadcast, ModelInfo}
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.{MklDnnLayer, TensorMMap}
import com.intel.analytics.bigdl.nn.tf.Const
import com.intel.analytics.bigdl.tensor.{QuantizedTensor, QuantizedType, Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.utils.Engine

import com.intel.analytics.bigdl.utils.intermediate.IRGraph
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.tfpark.Util._
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class TFModelBroadcast[T: ClassTag]()
                                   (implicit ev: TensorNumeric[T]) extends ModelBroadcast[T] {
//  private type NativeType = (String, (Array[TensorMMap], Array[TensorMMap]))
  private var broadcastModel: Broadcast[ModelInfo[T]] = _
  private var broadcastConsts: Broadcast[Map[String, Tensor[_]]] = _
//  private var broadcastParameters: Broadcast[Array[Tensor[T]]] = _
//  private var broadcastParametersNative: Broadcast[Array[NativeType]] = _
  private var nodeNumber: Int = _
  private var coreNumber: Int = _

  private def setNodeAndCore(): Unit = {
    nodeNumber = EngineRef.getNodeNumber()
    coreNumber = EngineRef.getCoreNumber()
  }

  /**
    * broadcast the model
    * first get and clear Const values from the model
    * then get and clear the weight and bias parameters from the model
    * finally broadcast Const values, the parameters and model(without parameters) separately
    *
    * @param sc    SparkContext
    * @param model model to broadcast
    * @return this
    */
  override def broadcast(sc: SparkContext, model: Module[T]): this.type = {
    CachedModels.deleteAll(uuid) // delete the models on driver


    // broadcast Consts
//    if (model.isInstanceOf[Container[_, _, T]]) {
//      val moduleConsts = getAndClearConsts(model.asInstanceOf[Container[_, _, T]])
//      // TODO: broadcast Const, model structure and weight in the same broadcast.
//      broadcastConsts = sc.broadcast(moduleConsts)
//    }
    // broadcast weight and model
    val weightsBias = getAndClearWeightBias(model.parameters())
    val extraParams = getAndClearExtraParameters(model.getExtraParameter())
    broadcastModel = sc.broadcast(ModelInfo[T](uuid, model))
    var i = 0
    while (i < model.parameters()._1.length){
      println("when broadcast")
      println(s"weights ${i} size: ${model.parameters()._1(i).size().mkString(",")}")
      i += 1
    }
    //      broadcastParameters = sc.broadcast(weightsBias)

    // For quantized model if we don't clone weightsBias, the original model will be released also
    // when we delete all models used in `ModelBroadcast`.
    putWeightBias(SerializationUtils.clone(weightsBias), model)
//    initGradWeightBias(weightsBias, model)
    putExtraParams(extraParams, model)

    setNodeAndCore()
    this
  }

  /**
    * get the broadcast model
    * put the weight and bias back to the model
    *
    * @param initGradient If create a tensor for gradient when fetch the model. Please note that
    *                     the gradient is not needed in model inference
    * @return model
    */
  override def value(initGradient: Boolean = false, shareWeight: Boolean = true): Module[T] = {
    EngineRef.setCoreNumber(coreNumber)
//    Engine.setNodeAndCore(nodeNumber, coreNumber)
    CachedModels.deleteAll(this.uuid)

    val localModel = broadcastModel.value.model.cloneModule()
    val uuid = broadcastModel.value.uuid
    CachedModels.add(uuid, localModel)


    // share Consts
//    if (localModel.isInstanceOf[Container[_, _, T]] && broadcastConsts.value.nonEmpty) {
//      putConsts(localModel.asInstanceOf[Container[_, _, T]], broadcastConsts.value)
//    }

    localModel
  }
}

private[zoo] class ModelInfo[T: ClassTag](val uuid: String, @transient var model: Module[T])(
  implicit ev: TensorNumeric[T]) extends Serializable {
  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    out.defaultWriteObject()
    val cloned = model.cloneModule()
    out.writeObject(cloned)
    CachedModels.add(uuid, cloned)
  }

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
    in.defaultReadObject()
    model = in.readObject().asInstanceOf[Module[T]]
    CachedModels.add(uuid, model)
  }
}


private[zoo] object ModelInfo {
  def apply[T: ClassTag](uuid: String, model: Module[T])(
    implicit ev: TensorNumeric[T]): ModelInfo[T] = new ModelInfo[T](uuid, model)
}


private[zoo] object CachedModels {

  import java.util.concurrent.ConcurrentHashMap

  import scala.collection._
  import scala.collection.convert.decorateAsScala._
  import scala.language.existentials

  type Modles = ArrayBuffer[Module[_]]

  private val cachedModels: concurrent.Map[String, Modles] =
    new ConcurrentHashMap[String, Modles]().asScala

  def add[T: ClassTag](uuid: String, model: Module[T])(implicit ev: TensorNumeric[T]): Unit =
    CachedModels.synchronized {
      val models = cachedModels.get(uuid) match {
        case Some(values) => values += model.asInstanceOf[Module[_]]
        case _ => ArrayBuffer(model.asInstanceOf[Module[_]])
      }
      cachedModels.put(uuid, models.asInstanceOf[Modles])
    }

  def deleteAll[T: ClassTag](currentKey: String)(implicit ev: TensorNumeric[T]): Unit =
    CachedModels.synchronized {
      val keys = cachedModels.keys
      for (key <- keys) {
        if (key != currentKey) {
          val models = cachedModels(key)
          for (model <- models) {
            model.release()
          }
          cachedModels.remove(key)
        }
      }
    }

  def deleteKey[T: ClassTag](key: String)(implicit ev: TensorNumeric[T]): Unit =
    CachedModels.synchronized {
      val keys = cachedModels.keys
      for (k <- keys) {
        if (k == key) {
          val models = cachedModels(key)
          for (model <- models) {
            model.release()
          }
          cachedModels.remove(key)
        }
      }
    }
}

object Util {

  private[zoo] def getAndClearWeightBias[T: ClassTag]
  (parameters: (Array[Tensor[T]], Array[Tensor[T]]))(implicit ev: TensorNumeric[T])
  : Array[Tensor[T]] = {
    clearTensor(parameters._2)
    getAndClearParameters(parameters._1)
  }

  private[zoo] def getAndClearExtraParameters[T: ClassTag]
  (parameters: Array[Tensor[T]])(implicit ev: TensorNumeric[T])
  : Array[Tensor[T]] = {
    getAndClearParameters(parameters)
  }

  private[zoo] def getAndClearParameters[T: ClassTag]
  (parameters: Array[Tensor[T]])(implicit ev: TensorNumeric[T])
  : Array[Tensor[T]] = {
    if (parameters != null) {
      if (parameters.length != 0) {
        var i = 0
        val retParams = new Array[Tensor[T]](parameters.length)
        //      val isQuantized = parameters._1.exists(_.getTensorType == QuantizedType)
        val (isCompacted, storage) = {
          val storage = Storage(parameters(0).storage.array())
          (parameters.map(_.nElement()).sum == storage.length(), storage)
        }


        // get extra parameters
        while (i < parameters.length) {
          if (parameters(i) != null) {
            val wb = parameters(i)
            retParams(i) = if (isCompacted) {
              Tensor[T](storage, wb.storageOffset(), wb.size(), wb.stride())
            } else {
              Tensor[T](Storage(wb.storage().array()), wb.storageOffset(), wb.size(), wb.stride())
            }
            //          wb.getTensorType match {
            //            case QuantizedType =>
            //              val quantTensor = wb.asInstanceOf[QuantizedTensor[T]]
            //              weightsBias(i) = QuantizedTensor[T](quantTensor.getStorage, quantTensor.maxOfRow,
            //                quantTensor.minOfRow, quantTensor.sumOfRow, quantTensor.size(), quantTensor.params)
            //            case _ =>
            //              weightsBias(i) = if (isCompacted) {
            //                Tensor[T](storage, wb.storageOffset(), wb.size(), wb.stride())
            //              } else {
            //                Tensor[T](Storage(wb.storage().array()), wb.storageOffset(), wb.size(), wb.stride())
            //              }
            //          }
            i += 1
          }
        }
        // clear parameters
        clearTensor(parameters)
        i = 0
        while (i < parameters.length){
          println(s"cleared weights ${i} size: ${parameters(i).size().mkString(",")}")
          i += 1
        }

        retParams
      } else {
        // just return an empty array when parameters is empty.
        Array()
      }
    } else {
      null
    }
  }


//  private[bigdl] def clearParamsAndExtraParams[T: ClassTag]
//  (parameters: (Array[Tensor[T]], Array[Tensor[T]]), extraParameters: Array[Tensor[T])(implicit ev: TensorNumeric[T])
//  : Unit = {
//    // clear parameters
//    clearTensor(parameters._1)
//    clearTensor(parameters._2)
//    clearTensor(extraParameters)
//  }

  private def clearTensor[T: ClassTag](tensors: Array[Tensor[T]])
                                      (implicit ev: TensorNumeric[T]): Unit = {
    if (tensors != null) {
      var i = 0
      while (i < tensors.length) {
        if (tensors(i) != null) {
          tensors(i).set()
        }
        i += 1
      }
    }
  }

  private[zoo] def putWeightBias[T: ClassTag](
                                               broadcastWeightBias: Array[Tensor[T]],
                                               localModel: Module[T])(implicit ev: TensorNumeric[T]): Unit = {
    val localWeightBias = localModel.parameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        clearAndSet(localWeightBias(i), broadcastWeightBias(i))
      }
      i += 1
    }

    def clearAndSet(old: Tensor[T], other: Tensor[T]): Unit = {
      //      if (old.getTensorType == QuantizedType && other.getTensorType == QuantizedType) {
      //        val quantOld = old.asInstanceOf[QuantizedTensor[T]]
      //        val quantOther = other.asInstanceOf[QuantizedTensor[T]]
      //
      //        if (quantOld.getNativeStorage != quantOther.getNativeStorage) {
      //          quantOld.release()
      //        }
      //      }

      old.set(other)
    }
  }

  private[zoo] def putExtraParams[T: ClassTag](
                                                broadcastExtraParams: Array[Tensor[T]],
                                                localModel: Module[T])(implicit ev: TensorNumeric[T]): Unit = {
    val localExtraParams = localModel.getExtraParameter()
    if (localExtraParams != null) {
      var i = 0
      while (i < localExtraParams.length) {
        if (localExtraParams(i) != null) {
          localExtraParams(i).set(broadcastExtraParams(i))

        }
        i += 1
      }
    }

  }

  private[zoo] def initGradWeightBias[T: ClassTag](
                                                    broadcastWeightBias: Array[Tensor[T]],
                                                    localModel: Module[T])(implicit ev: TensorNumeric[T]): Unit = {
    val (localWeightBias, localGradWeightBias) = localModel.parameters()
    // init gradient with a compacted storage
    val storage = Storage[T](localGradWeightBias.map(_.nElement()).sum)
    val isQuantized = broadcastWeightBias.exists(_.getTensorType == QuantizedType)
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        val wb = broadcastWeightBias(i)
        wb.getTensorType match {
          case QuantizedType =>
            localGradWeightBias(i).set(Tensor(1))
          case _ =>
            localGradWeightBias(i).set(storage, wb.storageOffset(), wb.size(), wb.stride())
        }
      }
      i += 1
    }
  }

//  private[zoo] def getAndClearConsts[T: ClassTag](
//                                                   model: Container[_, _, T])(implicit ev: TensorNumeric[T]): Map[String, Tensor[_]] = {
//    val moduleConsts = model.findModules("Const")
//      .map(_.asInstanceOf[Const[T, _]])
//      .map(v => (v, v.value.shallowClone()))
//    moduleConsts.foreach(_._1.value.set())
//    val result = moduleConsts.map(v => (v._1.getName(), v._2)).toMap[String, Tensor[_]]
//    require(result.size == moduleConsts.length, s"${model}'s Const node's name is duplicated," +
//      s"please check your model.")
//    result
//  }

//  private[zoo] def putConsts[T: ClassTag](
//                                           model: Container[_, _, T],
//                                           consts: Map[String, Tensor[_]])(implicit ev: TensorNumeric[T]): Unit = {
//    val moduleConsts = model.findModules("Const")
//      .map(_.asInstanceOf[Const[T, _]])
//    moduleConsts.foreach { const =>
//      val constValue = const.value.asInstanceOf[NumericWildcard]
//      val constName = const.getName()
//      constValue.asInstanceOf[Tensor[NumericWildcard]]
//        .set(consts(constName).asInstanceOf[Tensor[NumericWildcard]])
//    }
//  }

}
