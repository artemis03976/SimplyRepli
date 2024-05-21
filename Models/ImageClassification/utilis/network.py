def get_network_cfg(network_cfg, network):
    if network in network_cfg.keys():
        return network_cfg[network]
    else:
        raise NotImplementedError('Unsupported model: {}'.format(network))
