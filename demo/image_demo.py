from argparse import ArgumentParser

from mmcls.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default='demo/demo.JPEG',help='Image file')
    parser.add_argument('--config', default='/lustre/S/jiangfei/BinaryNeuralNetwork/configs/baseline/modeltest.py',help='Config file')
    parser.add_argument('--checkpoint', default='/lustre/S/jiangfei/BinaryNeuralNetwork/work_dirs/react_A_32/a100_s1/epoch_200.pth',help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    #result = inference_model(model, args.img)
    print(result)
    # # show the results
    show_result_pyplot(model, args.img, result)
 


if __name__ == '__main__':
    main()
