import copy
import warnings
import ttnn
from mmcv.utils import build_from_cfg, ConfigDict, deprecated_api_warning
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.registry import ATTENTION, POSITIONAL_ENCODING, ACTIVATION_LAYERS, FEEDFORWARD_NETWORK, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from bos_metal import op


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)


def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer layer."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)


@ACTIVATION_LAYERS.register_module(name="ReLU_tt", force=True)
class ReLU(op.BaseModule):
    def __init__(self, inplace=False):
        super().__init__(requires_shape=False)

    def forward(self, inputs, memory_config=None):
        return ttnn.relu(inputs, memory_config=memory_config)


@ACTIVATION_LAYERS.register_module(name="Sigmoid_tt", force=True)
class Sigmoid(op.BaseModule):
    def __init__(self, inplace=False):
        super().__init__(requires_shape=False)

    def forward(self, inputs, memory_config=None):
        return ttnn.sigmoid(inputs, memory_config=memory_config)


@ACTIVATION_LAYERS.register_module(name="GELU_tt", force=True)
class GELU(op.BaseModule):
    def __init__(self, inplace=False):
        super().__init__(requires_shape=False)

    def forward(self, inputs, memory_config=None):
        return ttnn.gelu(inputs, memory_config=memory_config)
    

from collections.abc import MutableMapping
class MyDict(MutableMapping):
    """
    A dictionary-like container that

    * never raises KeyError on chained lookup – missing items are auto-created
      as empty MyDict placeholders;
    * can be constructed from ordinary (possibly nested) dicts;
    * wraps every stored value so that `node.value` returns the real payload
      (or ``None`` for placeholders created implicitly).
    """

    # --------------------------------------------------------------------- #
    # core construction helpers                                             #
    # --------------------------------------------------------------------- #
    __slots__ = ("_data", "_has_payload", "_payload")

    def __init__(self, initial=None, *, _has_payload=False, _payload=None):
        self._data: dict[str, "MyDict"] = {}
        self._has_payload: bool = _has_payload
        self._payload = _payload

        # If the user passed a plain mapping, recursively convert its leaves
        if isinstance(initial, dict):
            for k, v in initial.items():
                self[k] = v           # delegates to our __setitem__
        elif initial is not None:
            # Treat any non-dict object as a leaf payload
            self._has_payload = True
            self._payload = initial

    # --------------------------------------------------------------------- #
    # MutableMapping protocol                                               #
    # --------------------------------------------------------------------- #
    def __getitem__(self, key):
        if self._has_payload:
            # You’re trying to index into a leaf node.  Instead of failing,
            # promote a child placeholder so that the chain can continue.
            self._has_payload = False
            self._payload = None

        if key not in self._data:
            # Create and store a *placeholder* that reports .value == None
            self._data[key] = MyDict()
        return self._data[key]

    def __setitem__(self, key, value):
        # Any plain dict becomes a nested MyDict; everything else
        # (numbers, lists, custom objects, …) becomes an **owned payload**
        if isinstance(value, dict) and not isinstance(value, MyDict):
            wrapped = MyDict(value)
        elif isinstance(value, MyDict):
            wrapped = value
        else:
            wrapped = MyDict(_has_payload=True, _payload=value)

        self._data[key] = wrapped

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    # --------------------------------------------------------------------- #
    # Convenience helpers                                                   #
    # --------------------------------------------------------------------- #
    @property
    def value(self):
        """
        * ``None``  → this node was auto-created during a missing-key lookup.
        * anything else → the real object stored at this leaf.
        """
        return self._payload if self._has_payload else None

    # Pretty representation (helps with debugging / printing)
    def __repr__(self):
        if self._has_payload:
            return f"MyDict(value={self._payload!r})"
        return f"MyDict({self._data!r})"
    
    
@FEEDFORWARD_NETWORK.register_module(force=True)
class FFN_tt(op.BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    @deprecated_api_warning(
        {"dropout": "ffn_drop", "add_residual": "add_identity"}, cls_name="FFN_tt"
    )
    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU_tt", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    op.Linear(in_channels, feedforward_channels),
                    self.activate,
                    op.Identity(requires_shape=False),
                )
            )
            in_channels = feedforward_channels
        layers.append(op.Linear(feedforward_channels, embed_dims))
        layers.append(op.Identity(requires_shape=False))
        self.layers = Sequential(*layers)
        self.dropout_layer = op.Identity(requires_shape=False)
        self.add_identity = add_identity
        self.add = op.Add(requires_shape=False, deallocate_input=True)

    @deprecated_api_warning({"residual": "identity"}, cls_name="FFN")
    def forward(self, x, identity=None, memory_config=MyDict(), program_config=MyDict(), inplace=True, **kwargs):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers[0][0](
            x, 
            memory_config=memory_config.value,
            program_config=program_config[0].value
        )
        
        if program_config[0].value is None:
            out = self.activate(out)
        
        for idx, layer in enumerate(self.layers):
            if idx == 0: continue
            if idx == self.num_fcs - 1: break
            
            out = layer[0](
                out,
                memory_config=memory_config.value,
                program_config=program_config[idx].value,
            )
            
            if program_config[idx].value is None:
                out = self.activate(out)
                
        out = self.layers[idx](
            out,
            memory_config=memory_config.value,
            program_config=program_config[idx].value,
        )
        
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        
        if inplace: 
            return ttnn.add_(identity, out)
        
        return ttnn.add(identity, out, memory_config=identity.memory_config())
    
    
class BaseTransformerLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN_tt",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU_tt", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN_tt"),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):

        deprecated_args = dict(
            feedforward_channels="feedforward_channels",
            ffn_dropout="ffn_drop",
            ffn_num_fcs="num_fcs",
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments "
                    f"to a dict named `ffn_cfgs`. ",
                    DeprecationWarning,
                )
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & {"self_attn", "norm", "ffn", "cross_attn"} == set(
            operation_order
        ), (
            f"The operation_order of"
            f" {self.__class__.__name__} should "
            f"contains all four operation type "
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"
        )

        num_attn = operation_order.count("self_attn") + operation_order.count(
            "cross_attn"
        )
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count("ffn")
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if "embed_dims" not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]["embed_dims"] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]["embed_dims"] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index], dict(type="FFN_tt"))
            )

        self.norms = ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        memory_config=MyDict(),
        program_config=MyDict(),
        inplace=True,
        **kwargs,
    ):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        if memory_config['q_input'].value is not None:
            if query.memory_config() != memory_config['q_input'].value:
                query = ttnn.to_memory_config(query, memory_config['q_input'].value)
                if query_pos is not None:
                    query_pos = ttnn.to_memory_config(query_pos, query.memory_config())
                
                if memory_config['kv_input'].value is None:
                    memory_config['kv_input'] = memory_config['q_input'].value
                
                if key is not None: 
                    key = ttnn.to_memory_config(key, memory_config['kv_input'].value)
                    if key_pos is not None:
                        key_pos = ttnn.to_memory_config(key_pos, key.memory_config())
                    value = ttnn.to_memory_config(value, memory_config['kv_input'].value) 

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )
        
        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query, temp_key, temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    memory_config=memory_config['self_attn'],
                    program_config=program_config['self_attn'],
                    inplace=inplace,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](
                    query, 
                    memory_config=memory_config['norm'].value,
                    program_config=program_config['norm'].value
                )
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query, key, value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    memory_config=memory_config['cross_attn'],
                    program_config=program_config['cross_attn'],
                    inplace=inplace,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](
                    query, 
                    identity if self.pre_norm else None,
                    memory_config=memory_config['ffn'],
                    program_config=program_config['ffn'],
                    inplace=inplace,
                )
                ffn_index += 1
                
        return query